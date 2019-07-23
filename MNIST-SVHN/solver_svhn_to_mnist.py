import os

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torchvision import transforms

from model import D1

from models.skip_svhn_to_mnist import skip
from mnist_classifier import Net
from utils.denoising_utils import *


class Solver(object):
    def __init__(self, config, svhn_loader, mnist_loader):
        self.config = config
        self.svhn_loader = svhn_loader
        self.mnist_loader = mnist_loader
        self.reg_noise_std = config.reg_noise_std
        self.figsize = config.figsize
        self.d1 = None      # Decoder network for MNIST
        self.unshared_optimizer = None
        self.d_optimizer = None
        self.net = None
        self.net_optimizer = None
        self.mse = None
        self.PLOT = config.PLOT
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.d_conv_dim = config.d_conv_dim
        self.train_iters = config.train_iters
        self.batch_size = config.batch_size
        self.lr_net = config.lr_net
        self.lr_d = config.lr_d
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        self.d1_load_path = os.path.join(config.load_path, "d1-" + str(config.load_iter) + ".pkl")
        self.net_load_path = os.path.join(config.load_path, "g22-" + str(config.load_iter) + ".pkl")
        self.build_model()

    def build_model(self):
        """Builds a generator and a discriminator."""

        self.d1 = D1(conv_dim=self.d_conv_dim, use_labels=False)
        self.d1.load_state_dict(torch.load(self.d1_load_path))
        self.d_optimizer = optim.Adam(list(self.d1.parameters()), self.lr_d, [self.beta1, self.beta2])

        self.net = skip(
            3, 3,
            num_channels_down=[64, 128],
            num_channels_up=[64, 128],
            num_channels_skip=[0, 0],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')

        self.net.load_state_dict(torch.load(self.net_load_path))
        self.net_optimizer = optim.Adam(list(self.net.parameters()), lr=self.lr_net)
        self.unshared_optimizer = optim.Adam(list(self.net.unshared_parameters()), self.lr_d,
                                             [self.beta1, self.beta2])

        self.mse = torch.nn.MSELoss()

        # s = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        # print('Number of params in the main network: %d' % s)

        if torch.cuda.is_available():
            self.d1.cuda()
            self.net.cuda()
            self.mse.cuda()

    def merge_images(self, sources, targets, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size)) + 1
        merged = np.zeros([3, row * h, row * w * 2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
        return merged.transpose(1, 2, 0)

    def to_var(self, x, volatile=False):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        if volatile:
            return Variable(x, volatile=True)
        return Variable(x)

    def to_no_grad_var(self, x):
        x = self.to_data(x, no_numpy=True)
        return self.to_var(x, volatile=True)

    def to_data(self, x, no_numpy=False):
        """Converts variable to numpy."""
        if torch.cuda.is_available():
            x = x.cpu()
        if no_numpy:
            return x.data
        return x.data.numpy()

    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.unshared_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.net_optimizer.zero_grad()

    def _compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def load_mnist_classifier(self):
        mnist_classifier = Net()
        mnist_classifier.load_state_dict(torch.load("mnist_classifier.pkl"))
        mnist_classifier.cuda()
        return mnist_classifier

    def saveim(self, data, path, PLOT=False):
        image_numpy = data.data[0].cpu().float().numpy()
        image_numpy = image_numpy * 0.5 + 0.5
        plot_image_grid([np.clip(image_numpy, 0, 1)], path, PLOT, factor=self.figsize, nrow=1, print_path=False)

    def train(self):
        PLOT = self.PLOT

        svhn_iter = iter(self.svhn_loader)
        mnist_iter = iter(self.mnist_loader)

        # fixed svhn for sampling
        svhn_data, s_labels_data = svhn_iter.next()
        svhn, s_labels = self.to_var(svhn_data), self.to_var(s_labels_data)
        print('SVHN label is: ', s_labels.cpu().detach().numpy()[0])

        # Plot image
        img = svhn_data.squeeze()
        img_pil = transforms.ToPILImage()(img)
        img_np = pil_to_np(img_pil)

        path = os.path.join(self.sample_path, 'pic_svhn.png')
        plot_image_grid([img_np], path, PLOT, 4, 5)

        input_depth = img_np.shape[0]
        INPUT = 'noise'  # 'meshgrid'
        dtype = torch.cuda.FloatTensor
        net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()
        img_torch = np_to_torch(img_np).type(dtype)

        for i in range(self.train_iters):
            if self.reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * self.reg_noise_std)

            # load mnist dataset
            mnist_data, m_labels_data = mnist_iter.next()
            mnist, m_labels = self.to_var(mnist_data), self.to_var(m_labels_data)

            # ============ train D ============#
            # train with real images
            self.reset_grad()
            out = self.d1(mnist)
            d1_loss = torch.mean((out - 1) ** 2)

            d_real_loss = d1_loss
            d_real_loss.backward()
            self.d_optimizer.step()

            # train with fake images
            self.reset_grad()
            fake_mnist = self.net(net_input, mnist=True) #self.net
            out = self.d1(fake_mnist)
            d1_loss = torch.mean(out ** 2)

            d_fake_loss = d1_loss
            d_fake_loss.backward()
            self.d_optimizer.step()

            # ============ train G Net ============#
            # train with SVHN - DIP loss
            self.reset_grad()
            out = self.net(net_input)

            total_loss = self.mse(out, img_torch)
            total_loss.backward()
            self.net_optimizer.step()

            # train with MNIST - discriminator loss
            self.reset_grad()
            fake_mnist = self.net(net_input, mnist=True)
            out_mnist = self.d1(fake_mnist)
            total_loss = torch.mean((out_mnist - 1) ** 2)

            total_loss.backward()
            if i % 100 == 0:
                self.unshared_optimizer.step()

            # save and plot images
            if i % self.sample_step == 0:
                out_np = torch_to_np(out)
                path = os.path.join(self.sample_path, 'sample-%d-s-m-s.png' % i)
                plot_image_grid([np.clip(out_np, 0, 1), img_np], path, PLOT, factor=self.figsize, nrow=1)

                fake_mnist_np = torch_to_np(fake_mnist)
                path = os.path.join(self.sample_path, 'sample-%d-s-m-m.png' % i)
                plot_image_grid([np.clip(fake_mnist_np, 0, 1)], path, PLOT, factor=self.figsize, nrow=1)

    def test(self):
        n_test = 500
        print("N test per model: " + str(n_test))

        results = []
        scores = []
        mnist_classifier = self.load_mnist_classifier()
        test_iter = iter(self.svhn_loader)
        mnist_iter = iter(self.mnist_loader)
        output_dir = self.config.test_path
        for step in range(n_test):
            self.build_model()                          # reset model
            if (step + 2) * self.train_iters > len(self.mnist_loader):         # reset mnist
                mnist_iter = iter(self.mnist_loader)
            svhn_var, s_labels = next(test_iter)
            real_label = s_labels.cuda()
            img = svhn_var.squeeze()
            img_pil = transforms.ToPILImage()(img)
            img_np = pil_to_np(img_pil)
            dtype = torch.cuda.FloatTensor
            net_input = get_noise(3, 'noise', (self.config.image_size, self.config.image_size)).type(dtype).detach()
            net_input_saved = net_input.detach().clone()
            noise = net_input.detach().clone()
            img_torch = np_to_torch(img_np).type(dtype)

            for i in range(self.train_iters + 1):
                if self.reg_noise_std > 0:
                    net_input = net_input_saved + (noise.normal_() * self.reg_noise_std)

                # load mnist dataset
                mnist_data, m_labels_data = mnist_iter.next()
                mnist, m_labels = self.to_var(mnist_data), self.to_var(m_labels_data)

                # ============ train D ============#
                # train with real images
                self.reset_grad()
                out = self.d1(mnist)
                d1_loss = torch.mean((out - 1) ** 2)

                d_real_loss = d1_loss
                d_real_loss.backward()
                self.d_optimizer.step()

                # train with fake images
                self.reset_grad()
                fake_mnist = self.net(net_input, mnist=True)  # self.net
                out = self.d1(fake_mnist)
                d1_loss = torch.mean(out ** 2)

                d_fake_loss = d1_loss
                d_fake_loss.backward()
                self.d_optimizer.step()

                # ============ train G Net ============#
                # train with SVHN - DIP loss
                self.reset_grad()
                out = self.net(net_input)

                total_loss = self.mse(out, img_torch)
                total_loss.backward()
                self.net_optimizer.step()

                # train with MNIST - discriminator loss
                self.reset_grad()
                fake_mnist = self.net(net_input, mnist=True)
                out_mnist = self.d1(fake_mnist)
                total_loss = torch.mean((out_mnist - 1) ** 2)

                total_loss.backward()
                if i % 100 == 0:
                    self.unshared_optimizer.step()

            fake_mnist_var = self.net(net_input, mnist=True)
            fake_label = mnist_classifier(fake_mnist_var)
            pred = fake_label[0].data.max(-1)[1]
            score = 1.0 * self.to_data(pred.eq(real_label)).sum()
            scores.append(score)
            # == save im
            self.saveim(svhn_var, os.path.join(output_dir, "svhn_" + str(step) + ".png"))
            self.saveim(fake_mnist_var, os.path.join(output_dir, "mnist_" + str(step) + ".png"))
            print("step:", step, " real:", s_labels.item(), " pred:", pred.item(), " total:", np.sum(scores))
            # ==============

        results.append(np.mean(scores))
        results.append(np.std(scores))

        print("\n===== Final Results: =====")
        [print(str(100.0 * results[2 * i]) + ", " + str(results[2 * i + 1])) for i in range(len(results) // 2)]






