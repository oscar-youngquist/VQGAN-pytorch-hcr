import os
import numpy as np
import torch
from vqgan_lightning import LitVQGAN
from utils import ImagePaths
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

cwd = os.getcwd()
model_folder_path = os.path.join(cwd, "training_results", "03_22_202416_44_55")
model_file_path = os.path.join(model_folder_path, "lightning_logs", "version_0", "checkpoints", "epoch=12-step=1083342.ckpt")
negative_data_path = os.path.join(cwd, "..", "processed_data", "actual_negatives_eval.txt")

eval_output_path = os.path.join(model_folder_path, "actual_negative_images")

if not os.path.exists(eval_output_path):
    os.makedirs(eval_output_path)

# load negative image data
test_data = ImagePaths(negative_data_path, size=256)
negative_data_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

# load trained model
vqgan = LitVQGAN.load_from_checkpoint(model_file_path)

for batch_num, im in enumerate(negative_data_loader):
    recons, diff = vqgan.encode_GAT_nodes(im.cuda())

    im_np = torch.squeeze(im.cpu()).numpy().swapaxes(0,1).swapaxes(1,2)
    recons_np = torch.squeeze(recons.cpu()).numpy().swapaxes(0,1).swapaxes(1,2)
    diff_np = torch.squeeze(diff.cpu()).numpy().swapaxes(0,1).swapaxes(1,2)

    fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=True)
    
    axes[0].imshow(im_np)
    axes[0].axis('off')
    axes[0].set_title("Input")
    axes[1].imshow(recons_np)
    axes[1].axis('off')
    axes[1].set_title("Recon.")
    axes[2].imshow(diff_np)
    axes[2].axis('off')
    axes[2].set_title("Diff.")
    new_path = os.path.join(eval_output_path, "neg_eval_sample_{:d}.png".format(batch_num))
    batch_num += 1
    fig.savefig(new_path)
    plt.close()