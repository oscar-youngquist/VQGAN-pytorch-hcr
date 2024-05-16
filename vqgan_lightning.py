import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, weights_init
import lightning as L
import json
from datetime import datetime
import random

# define the LightningModule
class LitVQGAN(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)

        self.first_ever_validation_image = True

        # self.opt_vq, self.opt_disc = self.configure_optimizers(args)
        
        self.args = args

        # we want to save the hyperparameters we used
        self.save_hyperparameters()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        self.logged_validation_batch = False

        self.comp_graph_added = False

        self.validation_losses = []
        self.avg_validation_loss = None

    def training_step(self, batch, batch_idx):
        opt_vq, opt_disc = self.optimizers()

        if self.logged_validation_batch:
            self.logged_validation_batch = False

        # add the computational graph of the model
        if not self.comp_graph_added:
            self.comp_graph_added = True
            sample_img = torch.rand((1,3,256,256)).to(self.args.device)
            self.logger.experiment.add_graph(self.vqgan, sample_img)

        # training_step defines the train loop.
        decoded_images, _, q_loss = self.vqgan(batch)

        disc_real = self.discriminator(batch)
        disc_fake = self.discriminator(decoded_images)

        disc_factor = self.vqgan.adopt_weight(self.args.disc_factor, self.global_step, threshold=self.args.disc_start)

        perceptual_loss = self.perceptual_loss(batch, decoded_images)
        rec_loss = torch.abs(batch - decoded_images)
        perceptual_rec_loss = self.args.perceptual_loss_factor * perceptual_loss + self.args.rec_loss_factor * rec_loss
        perceptual_rec_loss = perceptual_rec_loss.mean()
        g_loss = -torch.mean(disc_fake)

        λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
        vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

        d_loss_real = torch.mean(F.relu(1. - disc_real))
        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
        gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

        opt_vq.zero_grad()
        # vq_loss.backward(retain_graph=True)
        self.manual_backward(vq_loss, retain_graph=True)
        

        # just completely skip updating the GAN disciminator unless it has "turned on"?
        # if disc_factor > 0.0:
        
        opt_disc.zero_grad()
        # gan_loss.backward()
        self.manual_backward(gan_loss)
        
        opt_vq.step()
        opt_disc.step()

        self.log_dict({"train/vq_loss":vq_loss, "train/gan_loss":gan_loss, "train/q_loss":q_loss, "train/g_loss":g_loss, "train/rec_loss":rec_loss.mean(), 
                       "train/perceptual_loss":perceptual_loss.mean(), "train/perceptual_rec_loss":perceptual_rec_loss,
                       "train/d_loss_real":d_loss_real, "train/d_loss_fake":d_loss_fake}, prog_bar=True)
        
        # randomly save off some images
        rand_sample = random.random()
        if rand_sample > 0.999:
            vutils.save_image(decoded_images, os.path.join(self.args.output_path, "training_images", f"{self.current_epoch}_{batch_idx}_reconstruction.png"), nrow=4)
            vutils.save_image(torch.abs(batch - decoded_images), os.path.join(self.args.output_path, "training_images", f"{self.current_epoch}_{batch_idx}_difference.png"), nrow=4)
            vutils.save_image(batch, os.path.join(self.args.output_path, "training_images", f"{self.current_epoch}_{batch_idx}_input.png"), nrow=4)

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        decoded_images, _, q_loss = self.vqgan(batch)
        
        disc_real = self.discriminator(batch)
        disc_fake = self.discriminator(decoded_images)

        disc_factor = self.vqgan.adopt_weight(self.args.disc_factor, self.global_step, threshold=self.args.disc_start)

        perceptual_loss = self.perceptual_loss(batch, decoded_images)
        rec_loss = torch.abs(batch - decoded_images)
        perceptual_rec_loss = self.args.perceptual_loss_factor * perceptual_loss + self.args.rec_loss_factor * rec_loss
        perceptual_rec_loss = perceptual_rec_loss.mean()

        d_loss_real = torch.mean(F.relu(1. - disc_real))
        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
        gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

        # Could also do this via a callback... might switch later
        if not self.logged_validation_batch:
            output_path = os.path.join(self.args.output_path, "validation_images", f"{self.current_epoch}_{batch_idx}.png")
            
            if self.first_ever_validation_image:
                self.first_ever_validation_image = False
                output_path = os.path.join(self.args.output_path, "validation_images", f"{self.current_epoch-1}_{batch_idx}.png")
            
            self.logged_validation_batch = True
            # save the images to file
            vutils.save_image(decoded_images, os.path.join(self.args.output_path, "validation_images", f"{self.current_epoch}_{batch_idx}_reconstruction.png"), nrow=4)
            vutils.save_image(torch.abs(batch - decoded_images), os.path.join(self.args.output_path, "validation_images", f"{self.current_epoch}_{batch_idx}_difference.png"), nrow=4)
            vutils.save_image(batch, os.path.join(self.args.output_path, "validation_images", f"{self.current_epoch}_{batch_idx}_input.png"), nrow=4)

            # log the images to tensorboard
            # need to loop over....
            for i in range(0, self.args.batch_size):
                self.logger.experiment.add_image("images/input_val_{:d}".format(i), batch[i], self.current_epoch)
                self.logger.experiment.add_image("images/recon_val_{:d}".format(i), decoded_images[i], self.current_epoch)
                self.logger.experiment.add_image("images/difference_val_{:d}".format(i), torch.abs(batch[i] - decoded_images[i]), self.current_epoch)


        # all of these are done per-batch
        self.log_dict({"val/gan_loss":gan_loss, "val/rec_loss":rec_loss.mean(), "val/q_loss":q_loss, 
                "val/perceptual_loss":perceptual_loss.mean(), "val/perceptual_rec_loss":perceptual_rec_loss,
                "val/d_loss_real":d_loss_real, "val/d_loss_fake":d_loss_fake}, prog_bar=True)
        
        self.validation_losses.append(perceptual_rec_loss.detach().float().cpu().numpy().tolist()) 
        
    def on_train_epoch_end(self):
        # add weight histograms
        self.histogram_adder()
    
    def on_validation_epoch_end(self):
        self.avg_validation_loss = np.mean(self.validation_losses)
        self.validation_losses.clear()

    def histogram_adder(self):
        # iterate over parameters and add their values to a histogram 
        for name, params in self.named_parameters():
            _name = "weights/" + name
            self.logger.experiment.add_histogram(_name, params, self.current_epoch)


    def get_discrim_features(self, x_batch, recons_batch):
        _ = self.discriminator(x_batch)
        input_activations = self.discriminator.get_last_layer_activations()

        _ = self.discriminator(recons_batch)
        recon_activations = self.discriminator.get_last_layer_activations()

        return torch.abs(input_activations - recon_activations), input_activations
    
    def encode_GAT_nodes(self, node_batch):
        _cached_mode = self.training
        self.vqgan.eval()
        decoded_images = None
        with torch.no_grad():
            decoded_images, _, _ = self.vqgan(node_batch)

        # discrim_feats_diff, discrim_feats_input = self.get_discrim_features(node_batch, decoded_images)

        # , discrim_feats_diff, discrim_feats_input
        self.training = _cached_mode
        return decoded_images, torch.abs(node_batch - decoded_images)

    def configure_optimizers(self):
        lr = self.args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(self.args.beta1, self.args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(self.args.beta1, self.args.beta2))

        return opt_vq, opt_disc