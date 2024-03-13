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

    def training_step(self, batch, batch_idx):
        opt_vq, opt_disc = self.optimizers()

        if self.logged_validation_batch:
            self.logged_validation_batch = False

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

        opt_disc.zero_grad()
        # gan_loss.backward()
        self.manual_backward(gan_loss)

        opt_vq.step()
        opt_disc.step()

        self.log_dict({"train/vq_loss":vq_loss, "train/gan_loss":gan_loss, "train/rec_loss":rec_loss.mean(), 
                       "train/perceptual_loss":perceptual_loss.mean(), "train/perceptual_rec_loss":perceptual_rec_loss,
                       "train/d_loss_real":d_loss_real, "train/d_loss_fake":d_loss_fake}, prog_bar=True)
        
        # randomly save off some images
        rand_sample = random.random()
        if rand_sample > 0.95:
            vutils.save_image(decoded_images, os.path.join(self.args.output_path, "training_images", f"{self.current_epoch}_{batch_idx}.jpg"), nrow=4)
            vutils.save_image(torch.abs(batch - decoded_images), os.path.join(self.args.output_path, "training_images", f"{self.current_epoch}_{batch_idx}_difference.jpg"), nrow=4)

    def validation_step(self, batch, batch_inx):
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
            output_path = os.path.join(self.args.output_path, "validation_images", f"{self.current_epoch}_{batch_inx}.jpg")
            
            if self.first_ever_validation_image:
                output_path = os.path.join(self.args.output_path, "validation_images", f"{self.current_epoch-1}_{batch_inx}.jpg")
            
            self.logged_validation_batch = True
            vutils.save_image(decoded_images, os.path.join(self.args.output_path, "validation_images", f"{self.current_epoch}_{batch_inx}.jpg"), nrow=4)
            vutils.save_image(torch.abs(batch - decoded_images), os.path.join(self.args.output_path, "validation_images", f"{self.current_epoch}_{batch_inx}_difference.jpg"), nrow=4)

        self.log_dict({"val/gan_loss":gan_loss, "val/rec_loss":rec_loss.mean(),
                "val/perceptual_loss":perceptual_loss.mean(), "val/perceptual_rec_loss":perceptual_rec_loss,
                "val/d_loss_real":d_loss_real, "val/d_loss_fake":d_loss_fake}, prog_bar=True)



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