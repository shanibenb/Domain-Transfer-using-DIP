import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.resnet import ResNet_decoders
import numpy as np
from models.skip import skip


class DIP_AE_Model(BaseModel):
    def name(self):
        return 'DIP_AE_Model'

    def initialize(self, opt):
        input_depth = opt.input_nc
        output_depth = opt.output_nc
        BaseModel.initialize(self, opt)
        self.net_shared = skip(input_depth, num_channels_down = [64, 128, 256, 256, 256],
                        num_channels_up   = [64, 128, 256, 256, 256],
                        num_channels_skip = [4, 4, 4, 4, 4],
                        upsample_mode=['nearest', 'nearest', 'bilinear', 'bilinear', 'bilinear'],
                        need_sigmoid=True, need_bias=True, pad='reflection')
        self.netDec_b = ResNet_decoders(opt.ngf, output_depth)


        self.net_input = self.get_noise(self.opt.input_nc, 'noise', (self.opt.fineSize, self.opt.fineSize))
        self.net_input_saved = self.net_input.detach().clone()
        self.noise = self.net_input.detach().clone()

        use_sigmoid = opt.no_lsgan
        self.netD = networks.define_D(opt.output_nc, opt.ndf,
                                        opt.which_model_netD,
                                        opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if len(self.gpu_ids) > 0:
            dtype = torch.cuda.FloatTensor
            self.net_input = self.net_input.type(dtype).detach()
            self.net_shared = self.net_shared.type(dtype)
            self.netDec_b = self.netDec_b.type(dtype)
            self.netD = self.netD.type(dtype)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netDec_b, 'Dec_b', which_epoch)
            self.load_network(self.net_shared, 'Net_shared', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            self.fake_B_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.mse = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizer_Net = torch.optim.Adam(
                itertools.chain(self.net_shared.parameters(), self.netDec_b.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_b = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_Net)
            self.optimizers.append(self.optimizer_D_b)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.net_shared)
        networks.print_network(self.netDec_b)
        networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        # 'A' is given as single_dataset
        input_B = input['A']
        if len(self.gpu_ids) > 0:
            input_B = input_B.cuda(self.gpu_ids[0])
        self.input_B = input_B
        # 'A' is given as single_dataset
        self.image_paths = input['A_paths']

    def fill_noise(self, x, noise_type):
        """Fills tensor `x` with noise of type `noise_type`."""
        if noise_type == 'u':
            x.uniform_()
        elif noise_type == 'n':
            x.normal_()
        else:
            assert False

    def np_to_torch(self, img_np):
        '''Converts image in numpy.array to torch.Tensor.

        From C x W x H [0..1] to  C x W x H [0..1]
        '''
        return torch.from_numpy(img_np)[None, :]

    def get_noise(self, input_depth, method, spatial_size, noise_type='u', var=1. / 10):
        """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
        initialized in a specific way.
        Args:
            input_depth: number of channels in the tensor
            method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
            spatial_size: spatial size of the tensor to initialize
            noise_type: 'u' for uniform; 'n' for normal
            var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
        """
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        if method == 'noise':
            shape = [1, input_depth, spatial_size[0], spatial_size[1]]
            net_input = torch.zeros(shape)

            self.fill_noise(net_input, noise_type)
            net_input *= var
        elif method == 'meshgrid':
            assert input_depth == 2
            X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                               np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
            meshgrid = np.concatenate([X[None, :], Y[None, :]])
            net_input = self.np_to_torch(meshgrid)
        else:
            assert False

        return net_input

    def forward(self):
        self.real_B = Variable(self.input_B)

    def test(self):
        real_B = Variable(self.input_B, volatile=True)
        fake_B = self.netDec(self.netEnc(real_B))
        self.fake_B = fake_B.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D = self.backward_D_basic(self.netD, self.real_B, fake_B)
        self.loss_D = loss_D.item()

    def backward_G(self):
        # Train with B images - GAN loss
        fake_B = self.netDec_b(self.net_shared(self.net_input))
        pred_fake_B = self.netD(fake_B)
        loss_G = self.criterionGAN(pred_fake_B, True)

        loss_G.backward()

        self.fake_B = fake_B.data
        self.loss_G = loss_G.item()

    def optimize_parameters(self):
        # forward
        self.forward()

        # G
        self.optimizer_Net.zero_grad()
        self.backward_G()
        self.optimizer_Net.step()

        # D
        self.optimizer_D_b.zero_grad()
        self.backward_D()
        self.optimizer_D_b.step()


    def get_current_errors(self):
        ret_errors = OrderedDict(
            [('D', self.loss_D), ('G_B', self.loss_G) ])
        return ret_errors

    def get_current_visuals(self):
        real_B = util.tensor2im(self.input_B)
        fake_B = util.tensor2im(self.fake_B)

        ret_visuals = OrderedDict([('real_B', real_B), ('fake_B', fake_B), ])

        return ret_visuals

    def save(self, label):
        self.save_network(self.netDec_b, 'Dec_b', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        self.save_network(self.net_shared, 'Net_shared', label, self.gpu_ids)
