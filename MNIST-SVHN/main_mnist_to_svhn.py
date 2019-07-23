import argparse
import logging
from torch.backends import cudnn

from data_loader import get_loader
from solver_mnist_to_svhn import Solver
from utils.denoising_utils import *


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    svhn_loader, mnist_loader, svhn_test_loader, mnist_test_loader = get_loader(config)

    if config.mode == 'train':
        solver = Solver(config, svhn_loader, mnist_loader)
    elif config.mode == 'test':
        solver = Solver(config, svhn_test_loader, mnist_test_loader)
    cudnn.benchmark = True

    # create directories if not exist
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    if not os.path.exists(config.test_path):
        os.makedirs(config.test_path)

    base = config.log_path
    filename = os.path.join(base, str(config.max_items))
    if not os.path.isdir(base):
        os.mkdir(base)
    logging.basicConfig(filename=filename, level=logging.DEBUG)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=64)

    # training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--lr_d', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # misc
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--mnist_path', type=str, default='./mnist')
    parser.add_argument('--svhn_path', type=str, default='./svhn')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--shuffle', type=bool, default=True)

    parser.add_argument('--load_iter', type=int, default=20000)
    parser.add_argument('--sample_step', type=int, default=200)
    parser.add_argument('--sample_path', type=str, default='./results/samples_M-S')
    parser.add_argument('--test_path', type=str, default='./results/test_M-S')
    parser.add_argument('--load_path', type=str, default='./results/models_autoencoder_M_S')
    parser.add_argument('--log_path', type=str, default='./results/logs_DIP')
    parser.add_argument('--use_augmentation', default=False, type=bool)
    parser.add_argument('--max_items', type=int, default=1)
    parser.add_argument('--reg_noise_std', type=float, default=1./30.)
    parser.add_argument('--figsize', type=int, default=5)
    parser.add_argument('--PLOT', type=bool, default=False, help='choose True to plot after every sample_step')

    config = parser.parse_args()
    print(config)
    main(config)
