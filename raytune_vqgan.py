from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
from ray.tune.search import ConcurrencyLimiter

from training_vqgan_lightning import TrainVQGAN
import numpy as np
import pandas as pd
import os
import argparse

def objective(config):

    # this will train a model for some number of epochs...
    model_trainer = TrainVQGAN(config)

    results_dict = {}
    results_dict["recon_loss"] = model_trainer.vqgan_lightning_model.avg_validation_loss
    return results_dict

# NOTE - the disc-start sample space needs to be adjusted based on the number of iterations in the reduced batch-size
search_space = {"learning_rate": tune.loguniform(5e-6, 1e-4), "disc_start": tune.qloguniform(1.0e2, 3.0e4, 1), "disc_factor": tune.uniform(0.5, 1.0)}
algo = OptunaSearch()

def tune_VQGAN(args, num_samples=50, num_epochs=1):
    def train_VQGAN(config):
        
        # find keys we need to copy out of args
        config_keys = list(config.keys())

        for key in config_keys:
            if key == "disc_start":
                setattr(args, key, int(config[key]))
            else:
                setattr(args, key, config[key])
        
        # call the objective function
        return objective(args)
    
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)
    trainable_with_gpu = tune.with_resources(train_VQGAN, {"gpu": 1, "cpu":20})
    
    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
            metric="recon_loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
            scheduler=scheduler
        ),
        param_space=search_space,
    )

    return tuner.fit()

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
    parser.add_argument('--disc-start', type=int, default=750000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=0.8, help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--output-path', type=str, default='/vqgan_training', help='Location to save training data.')
    parser.add_argument('--num-workers', type=int, default=6, help='Number of workers for the dataloader (default: 6).')

    args = parser.parse_args()
    args.train_dataset_path = r"/home/oyoungquist/Research/CPFP/processed_data/reprocessed_positive_train_deep_acc_eval_vqgan.txt"
    args.val_dataset_path = r"/home/oyoungquist/Research/CPFP/processed_data/reprocessed_positive_vali_deep_acc_eval_vqgan.txt"
    args.output_path = r"/home/oyoungquist/Research/CPFP/VQGAN-pytorch-hcr/training_results/"

    results = tune_VQGAN(args=args)
    print("Best config is:", results.get_best_result().config)