import os

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torchvision import transforms

from model import D2

from models.skip_mnist_to_svhn import skip
from svhn_classifier import Model
from utils.denoising_utils import *


class Solver(object):
    def __init__(self, config, svhn_loader, mnist_loader):
        self.config = config
        self.svhn_loader = svhn_loader
        self.mnist_loader = mnist_loader
        self.reg_noise_std = config.reg_noise_std
        self.figsize = config.figsize
        self.d2 = None      # Decoder network for SVHN
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
        self.net_load_path = os.path.join(config.load_path, "g11-" + str(config.load_iter) + ".pkl")
        self.d2_load_path = os.path.join(config.load_path, "d2-" + str(config.load_iter) + ".pkl")
        self.build_model()

    def build_model(self):
        """Builds a generator and a discriminator."""
        self.d2 = D2(conv_dim=self.d_conv_dim, use_labels=False)
        self.d2.load_state_dict(torch.load(self.d2_load_path))
        self.d_optimizer = optim.Adam(list(self.d2.parameters()), self.lr_d, [self.beta1, self.beta2])

        self.net = skip(
            num_input_channels=1, num_output_channels=1,
            num_channels_down=[64, 128],
            num_channels_up=[64, 128],
            num_channels_skip=[0, 0],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')

        self.net.load_state_dict(torch.load(self.net_load_path))
        self.net_optimizer = optim.Adam(list(self.net.parameters()), lr=self.lr_net)
        self.unshared_optimizer = optim.Adam(list(self.net.unshared_parameters()), self.lr_net,
                                             [self.beta1, self.beta2])

        self.mse = torch.nn.MSELoss()

        # s = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        # print('Number of params in the main network: %d' % s)

        if torch.cuda.is_available():
            self.d2.cuda()
            self.net.cuda()
            self.mse.cuda()

    def merge_images(self, sources, targets):
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

    def load_svhn_classifier(self):
        svhn_classifier = Model()
        svhn_classifier.load_state_dict(torch.load("svhn_classifier.pkl"))
        svhn_classifier.cuda()
        return svhn_classifier

    def saveim(self, data, path):
        image_numpy = data.data[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * 0.5 + 0.5) * 255.0
        im = Image.fromarray(image_numpy.astype(np.uint8))
        im.save(path)

    def train(self):
        PLOT = self.PLOT

        svhn_iter = iter(self.svhn_loader)
        mnist_iter = iter(self.mnist_loader)

        # fixed mnist for sampling
        mnist_data, m_labels_data = mnist_iter.next()
        mnist, m_labels = self.to_var(mnist_data), self.to_var(m_labels_data)
        print('SVHN label is: ', m_labels.cpu().detach().numpy()[0])

        # Plot image
        img = mnist_data.squeeze().unsqueeze(0)
        img_pil = transforms.ToPILImage()(img)
        img_np = pil_to_np(img_pil)

        # plot_image_grid([img_np], 4, 5);
        path = os.path.join(self.sample_path, 'pic_mnist.png')
        plot_image_grid([img_np], path, PLOT, 4, 5)

        # create net input noise
        input_depth = img_np.shape[0]
        INPUT = 'noise'  # 'meshgrid'
        dtype = torch.cuda.FloatTensor
        net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()
        img_torch = np_to_torch(img_np).type(dtype)

        for i in range(self.train_iters + 1):
            if self.reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * self.reg_noise_std)

            # load svhn dataset
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
            fake_svhn = self.net(net_input, svhn=True)
            out = self.d2(fake_svhn)
            d2_loss = torch.mean(out ** 2)

            d_fake_loss = d2_loss
            d_fake_loss.backward()
            self.d_optimizer.step()

            # ============ train G Net ============#
            # train with MNIST - DIP loss
            self.reset_grad()
            out_mnist = self.net(net_input)

            total_loss = self.mse(out_mnist, img_torch)
            total_loss.backward()
            self.net_optimizer.step()

            # train with SVHN - discriminator loss
            self.reset_grad()
            fake_svhn = self.net(net_input, svhn=True)
            out_svhn = self.d2(fake_svhn)
            total_loss = torch.mean((out_svhn - 1) ** 2)

            total_loss.backward()
            if i % 100 == 0:
                self.unshared_optimizer.step()

            # save and plot images
            if i % self.sample_step == 0:
                out_np = torch_to_np(out_mnist)
                path = os.path.join(self.sample_path, 'sample-%d-m-s-m.png' % i)
                plot_image_grid([np.clip(out_np, 0, 1), img_np], path, PLOT, factor=self.figsize, nrow=1)

                fake_mnist_np = torch_to_np(fake_svhn)
                path = os.path.join(self.sample_path, 'sample-%d-m-s-s.png' % i)
                plot_image_grid([np.clip(fake_mnist_np, 0, 1)], path, PLOT, factor=self.figsize, nrow=1)

    def test(self):
        n_test = 500
        print("N test per model: " + str(n_test))

        results = []
        scores = []
        svhn_classifier = self.load_svhn_classifier()
        test_iter = iter(self.mnist_loader)
        svhn_iter = iter(self.svhn_loader)
        output_dir = self.config.test_path
        for step in range(n_test):
            self.build_model()                          # reset model
            if (step + 2) * self.train_iters > len(self.svhn_loader):         # reset svhn
                svhn_iter = iter(self.svhn_loader)
            mnist_var, m_labels = next(test_iter)
            real_label = m_labels.cuda()
            img = mnist_var.squeeze().unsqueeze(0)
            img_pil = transforms.ToPILImage()(img)
            img_np = pil_to_np(img_pil)
            dtype = torch.cuda.FloatTensor
            net_input = get_noise(1, 'noise', (self.config.image_size, self.config.image_size)).type(dtype).detach()
            net_input_saved = net_input.detach().clone()
            noise = net_input.detach().clone()
            img_torch = np_to_torch(img_np).type(dtype)

            for i in range(self.train_iters + 1):
                if self.reg_noise_std > 0:
                    net_input = net_input_saved + (noise.normal_() * self.reg_noise_std)

                # load svhn dataset
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
                fake_svhn = self.net(net_input, svhn=True)
                out = self.d2(fake_svhn)
                d2_loss = torch.mean(out ** 2)

                d_fake_loss = d2_loss
                d_fake_loss.backward()
                self.d_optimizer.step()

                # ============ train G Net ============#
                # train with MNIST - DIP loss
                self.reset_grad()
                out_mnist = self.net(net_input)

                total_loss = self.mse(out_mnist, img_torch)
                total_loss.backward()
                self.net_optimizer.step()

                # train with SVHN - discriminator loss
                self.reset_grad()
                fake_svhn = self.net(net_input, svhn=True)
                out_svhn = self.d2(fake_svhn)
                total_loss = torch.mean((out_svhn - 1) ** 2)

                total_loss.backward()
                if i % 100 == 0:
                    self.unshared_optimizer.step()

            fake_svhn_var = self.net(net_input, svhn=True)
            fake_label = svhn_classifier(fake_svhn_var)
            pred = fake_label[0].data.max(1)[1]
            score = 1.0 * self.to_data(pred.eq(real_label)).sum()
            scores.append(score)
            # == save im
            mnist_var = torch.cat(3 * [mnist_var], dim=1)
            self.saveim(mnist_var, os.path.join(output_dir, "mnist_" + str(step) + ".png"))
            self.saveim(fake_svhn_var, os.path.join(output_dir, "svhn_" + str(step) + ".png"))
            print("step:", step, " real:", m_labels.item(), " pred:", pred.item(), " total:", np.sum(scores))
            # ==============

        results.append(np.mean(scores))
        results.append(np.std(scores))

        print("\n===== Final Results: =====")
        [print(str(100.0 * results[2 * i]) + ", " + str(results[2 * i + 1])) for i in range(len(results) // 2)]
