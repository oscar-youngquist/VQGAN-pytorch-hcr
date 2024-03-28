"""
PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
"""

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(args.image_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)
        # self.output_layer = nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1)

        self.final_layer_activations_ = None

    def forward(self, x):
        # x = self.model(x)
        # self.final_layer_activations_ = x
        # return self.output_layer(x)

        return self.model(x)


    def get_last_layer_activations(self):
        if self.final_layer_activations_ == None:
            print("Tried to retrive discrim. activations before processing images!")
            exit(-1)
        return self.final_layer_activations_