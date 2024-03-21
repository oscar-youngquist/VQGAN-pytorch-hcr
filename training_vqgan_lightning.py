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
from vqgan_lightning import LitVQGAN
import json
from datetime import datetime
import re

class TrainVQGAN:
    def __init__(self, args):

        self.prepare_training(args)

        self.vqgan_lightning_model = LitVQGAN(args)

        self.train(args)

    def train(self, args):
        # checkpoint_path = os.path.join(args.output_path, "model_checkpoints")

        # if not os.path.exists(checkpoint_path):
        #     os.makedirs(checkpoint_path)

        train_dataloader, val_dataloader = load_data(args)

        trainer = L.Trainer(max_epochs=100, default_root_dir=args.output_path)
        trainer.fit(model=self.vqgan_lightning_model, train_dataloaders=train_dataloader, 
                    val_dataloaders=val_dataloader)
        
    @staticmethod
    def prepare_training(args):

        now = datetime.now() # current date and time
        date_time = now.strftime("%m/%d/%Y%H:%M:%S")

        date_time = re.sub('[^0-9a-zA-Z]+', '_', date_time)

        print(date_time)

        # make the output folder if it does not exist
        if not os.path.exists(os.path.join(args.output_path, date_time)):
            os.makedirs(os.path.join(args.output_path, date_time), exist_ok=True)
            args.output_path = os.path.join(args.output_path, date_time)

        print("Output path: {:s}".format(args.output_path))

        # save the passed in arguments as a json file
        #    Turns out PyTorch Lightning just does this for us
        # with open(os.path.join(args.output_path, "commandline_arguments.json"), "w+") as f:
        #     json.dump(args.__dict__, f, indent=2)

        # now create the validation images directory
        if not os.path.exists(os.path.join(args.output_path, "validation_images")):
            os.makedirs(os.path.join(args.output_path, "validation_images"), exist_ok=True)

        # now create the training images directory
        if not os.path.exists(os.path.join(args.output_path, "training_images")):
            os.makedirs(os.path.join(args.output_path, "training_images"), exist_ok=True)
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 1024)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--train-dataset-path', type=str, default='/train_data', help='Path to training data (default: /train_data)')
    parser.add_argument('--val-dataset-path', type=str, default='/val_data', help='Path to training data (default: /val_data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.5e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=200000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=0.8, help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--output-path', type=str, default='/vqgan_training', help='Location to save training data.')
    parser.add_argument('--num-workers', type=int, default=6, help='Number of workers for the dataloader (default: 6).')

    args = parser.parse_args()
    args.train_dataset_path = r"/home/oyoungquist/Research/CPFP/processed_data/reprocessed_positive_train_deep_acc_eval_vqgan.txt"
    args.val_dataset_path = r"/home/oyoungquist/Research/CPFP/processed_data/reprocessed_positive_vali_deep_acc_eval_vqgan.txt"
    args.output_path = r"/home/oyoungquist/Research/CPFP/VQGAN-pytorch-hcr/training_results/"

    train_vqgan = TrainVQGAN(args)



