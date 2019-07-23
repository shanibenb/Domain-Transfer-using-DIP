import os

import numpy as np
import torch
import scipy.io
from torch import optim
from torch.autograd import Variable

from model import D2
from models.skip_mnist_to_svhn import skip
from utils.denoising_utils import *


class Solver(object):
    def __init__(self, config, svhn_loader, mnist_loader):
        self.config = config
        self.svhn_loader = svhn_loader
        self.mnist_loader = mnist_loader
        self.g11 = None
        self.d2 = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.d_conv_dim = config.d_conv_dim
        self.train_iters = config.train_iters
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.g11_load_path = os.path.join(config.load_path, "g11-" + str(config.load_iter) + ".pkl")
        self.d2_load_path = os.path.join(config.load_path, "d2-" + str(config.load_iter) + ".pkl")
        self.build_model()

    def build_model(self):
        """Builds a generator and a discriminator."""
        self.g11 = skip(
            num_input_channels=1, num_output_channels=1,
            num_channels_down=[64, 128],
            num_channels_up=[64, 128],
            num_channels_skip=[0, 0],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')

        self.d2 = D2(conv_dim=self.d_conv_dim, use_labels=False)

        g_params = list(self.g11.parameters())
        d_params = list(self.d2.parameters())

        self.g_optimizer = optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(d_params, self.lr, [self.beta1, self.beta2])

        if self.config.continue_training:
            self.d2.load_state_dict(torch.load(self.d2_load_path))
            self.g11.load_state_dict(torch.load(self.g11_load_path))

        if torch.cuda.is_available():
            self.g11.cuda()
            self.d2.cuda()

    def merge_images(self, sources, targets):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([3, row * h, row * w * 2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
        return merged.transpose(1, 2, 0)

    def to_var(self, x):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def to_data(self, x):
        """Converts variable to numpy."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()

    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def _compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def train(self):
        dtype = torch.cuda.FloatTensor
        svhn_iter = iter(self.svhn_loader)
        mnist_iter = iter(self.mnist_loader)
        iter_per_epoch = min(len(svhn_iter), len(mnist_iter))
        input_depth = 1

        # fixed svhn for training
        fixed_svhn = self.to_var(svhn_iter.next()[0])

        # Train autoencoder for svhn
        for step in range(self.train_iters + 1):
            # reset data_iter for each epoch
            if (step + 1) % iter_per_epoch == 0:
                svhn_iter = iter(self.svhn_loader)

            # svhn dataset
            svhn_data, s_labels_data = svhn_iter.next()
            svhn, s_labels = self.to_var(svhn_data), self.to_var(s_labels_data).long().squeeze()

            # ============ train D ============#
            # train with real images
            self.reset_grad()
            out = self.d2(svhn)
            d2_loss = torch.mean((out - 1) ** 2)

            d_real_loss = d2_loss
            d_real_loss.backward()
            self.d_optimizer.step()

            # train with fake images
            self.reset_grad()
            svhn = get_noise(input_depth, 'noise', (self.config.image_size, self.config.image_size)).type(dtype).detach()
            fake_svhn = self.g11.forward(svhn, svhn=True)
            out = self.d2(fake_svhn)
            d1_loss = torch.mean(out ** 2)

            d_fake_loss = d1_loss
            d_fake_loss.backward()
            self.d_optimizer.step()

            # ============ train G ============
            self.reset_grad()
            svhn = get_noise(input_depth, 'noise', (self.config.image_size, self.config.image_size)).type(dtype).detach()
            fake_svhn = self.g11.forward(svhn, svhn=True)
            out = self.d2(fake_svhn)

            g_loss = torch.mean((out - 1) ** 2)
            g_loss.backward()
            self.g_optimizer.step()

            # print the log info
            if (step + 1) % self.log_step == 0:
                print('Step [%d/%d], d_real_loss: %.4f, d_fake_loss: %.4f, g_loss: %.4f'
                      % (step + 1, self.train_iters, d_real_loss.item(),
                         d_fake_loss.item(), g_loss.item()))

            # save the sampled images
            if (step + 1) % self.sample_step == 0:
                fixed_svhn_noise = get_noise(input_depth, 'noise', (self.config.image_size, self.config.image_size)).type(dtype).detach()
                fake_svhn = self.g11.forward(fixed_svhn_noise, svhn=True)
                svhn, fake_svhn = self.to_data(fixed_svhn), self.to_data(fake_svhn)

                merged = self.merge_images(svhn, fake_svhn)
                path = os.path.join(self.sample_path, 'sample-%d-m-s.png' % (step + 1))
                scipy.misc.imsave(path, merged)
                print('saved %s' % path)

            if (step + 1) % 10000 == 0:
                # save the model parameters for each epoch
                if self.config.continue_training:
                    g11_path = os.path.join(self.model_path, 'g11-%d.pkl' % (step + self.config.load_iter + 1))
                    d2_path = os.path.join(self.model_path, 'd2-%d.pkl' % (step + self.config.load_iter + 1))
                else:
                    g11_path = os.path.join(self.model_path, 'g11-%d.pkl' % (step + 1))
                    d2_path = os.path.join(self.model_path, 'd2-%d.pkl' % (step + 1))
                torch.save(self.g11.state_dict(), g11_path)
                torch.save(self.d2.state_dict(), d2_path)

