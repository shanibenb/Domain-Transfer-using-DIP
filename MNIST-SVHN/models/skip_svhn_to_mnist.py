import torch
import torch.nn as nn
from .common import *

class skip(nn.Module):
    """Assembles encoder-decoder with skip connections.
    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """

    def __init__(self, num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64], num_channels_up=[16, 32, 64], num_channels_skip=[4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True):

        super(skip, self).__init__()
        self.need_sigmoid = need_sigmoid
        self.n_scales = len(num_channels_down)

        if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
            upsample_mode = [upsample_mode] * self.n_scales
        if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
            downsample_mode = [downsample_mode] * self.n_scales

        if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
            filter_size_down = [filter_size_down] * self.n_scales

        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
            filter_size_up = [filter_size_up] * self.n_scales

        self.deeper0 = skipDeeper(num_input_channels, num_channels_down[0], filter_size_down[0], downsample_mode[0],
                                  need_bias, pad, act_fun)
        self.deeper1 = skipDeeper(num_channels_down[0], num_channels_down[1], filter_size_down[1], downsample_mode[1],
                                  need_bias, pad, act_fun)

        self.up0 = skipUp(num_channels_up[1], num_channels_up[0], filter_size_up[0],
                          need_bias, pad, act_fun, need1x1_up)
        self.up1 = skipUp(num_channels_down[1], num_channels_up[1], filter_size_up[1],
                          need_bias, pad, act_fun, need1x1_up)

        self.upsample0 = nn.Upsample(scale_factor=2, mode=upsample_mode[0])
        self.upsample1 = nn.Upsample(scale_factor=2, mode=upsample_mode[1])

        self.conv_last_MNIST = conv(num_channels_up[0], 1, 1, bias=need_bias, pad=pad)
        self.conv_last_SVHN = conv(num_channels_up[0], 3, 1, bias=need_bias, pad=pad)
        self.sigmoid_MNIST = nn.Sigmoid()
        self.sigmoid_SVHN = nn.Sigmoid()

    def forward(self, inputs, mnist=False):
        x = self.deeper0(inputs)
        x = self.deeper1(x)
        x = self.up1(self.upsample1(x))
        x = self.up0(self.upsample0(x))

        if mnist:
            out = self.conv_last_MNIST(x)
            if self.need_sigmoid:
                out = self.sigmoid_MNIST(out)
        else:
            out = self.conv_last_SVHN(x)
            if self.need_sigmoid:
                out = self.sigmoid_SVHN(out)

        return out

    def encode(self, inputs):
        x = self.deeper0(inputs)
        x = self.deeper1(x)
        x = self.deeper2(x)

        return x

    def encode_params(self):
        layers_basic = list(self.deeper0.parameters())
        layers_basic += list(self.deeper1.parameters())
        return layers_basic

    def decode_params(self):
        layers_basic = list(self.upsample1.parameters())
        layers_basic += list(self.upsample0.parameters())
        layers_basic += list(self.up1.parameters())
        layers_basic += list(self.up0.parameters())
        layers_basic += list(self.conv_last_SVHN.parameters()) + list(self.conv_last_MNIST.parameters())
        layers_basic += list(self.sigmoid_SVHN.parameters()) + list(self.sigmoid_MNIST.parameters())

        return layers_basic

    def unshared_parameters(self):
        return list(self.conv_last_SVHN.parameters()) + list(self.conv_last_MNIST.parameters()) + \
               list(self.sigmoid_SVHN.parameters()) + list(self.sigmoid_MNIST.parameters())


class skipDeeper(nn.Module):
    def __init__(self, input_depth, num_channels_down, filter_size_down, downsample_mode, need_bias, pad, act_fun):
        super(skipDeeper, self).__init__()
        self.deeper = nn.Sequential()
        self.deeper.add(conv(input_depth, num_channels_down, filter_size_down, 2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode))
        self.deeper.add(bn(num_channels_down))
        self.deeper.add(act(act_fun))

        self.deeper.add(conv(num_channels_down, num_channels_down, filter_size_down, bias=need_bias, pad=pad))
        self.deeper.add(bn(num_channels_down))
        self.deeper.add(act(act_fun))

    def forward(self, inputs):
        outputs = self.deeper(inputs)
        return outputs


class skipUp(nn.Module):
    def __init__(self, input_depth, num_channels_up, filter_size_up, need_bias, pad, act_fun, need1x1_up):
        super(skipUp, self).__init__()
        self.model = nn.Sequential()
        self.model.add(bn(input_depth))
        self.model.add(conv(input_depth, num_channels_up, filter_size_up, 1, bias=need_bias, pad=pad))
        self.model.add(bn(num_channels_up))
        self.model.add(act(act_fun))

        if need1x1_up:
            self.model.add(conv(num_channels_up, num_channels_up, 1, bias=need_bias, pad=pad))
            self.model.add(bn(num_channels_up))
            self.model.add(act(act_fun))

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs


