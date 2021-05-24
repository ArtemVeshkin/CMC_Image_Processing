import argparse
import sys
from PIL import Image
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.ndimage import shift
from skimage import io
import skimage
import matplotlib.pyplot as plt


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('blurred', nargs=1)
    parser.add_argument('kernel', nargs=1)
    parser.add_argument('output', nargs=1)
    parser.add_argument('noise_level', nargs=1)

    return parser


def make_blurred(clean, kernel, noise_level):
    kernel /= kernel.sum()
    blurred = convolve(clean, kernel) + np.random.normal(0, noise_level, clean.shape[:2])
    return blurred


def deblur(blurred, kernel, noise_level, iterations=100, alpha=0.1, beta=0.1, mu=0.1,
           iterations_folder='my_data/iterations/'):
    z = blurred
    v = np.zeros(blurred.shape)
    conv_kernel_t_u = 2 * convolve(blurred, kernel[::-1, ::-1])
    alpha = 0.1 if alpha < 0.1 else alpha

    for i in range(iterations):
        # if i % 5 == 0:
        #     print('Iteration ' + str(i))
        #     skimage.io.imsave(iterations_folder + 'iteration_' + str(i) + '.png', z.astype(np.uint8))

        grad = grad_conv(z, kernel, conv_kernel_t_u) + alpha * grad_tv(z)
        v = mu * v - grad
        z = z + beta * v
        z[z < 0] = 0
        z[z > 255] = 255
    return z


def grad_conv(z, kernel, conv_kernel_t_u):
    return 2 * convolve(convolve(z, kernel), kernel[::-1, ::-1]) - conv_kernel_t_u


def grad_tv(z):
    dx = np.sign(shift(z, (0, 1), mode='nearest') - z)
    dx = shift(dx, (0, -1), mode='nearest') - dx
    dy = np.sign(shift(z, (1, 0), mode='nearest') - z)
    dy = shift(dy, (-1, 0), mode='nearest') - dy
    return dx + dy


if __name__ == '__main__':
    # For testing
    generate = False
    if generate:
        noise_level = 5
        filename = 'avion'
        folder = 'my_data/'

        clean = np.array(Image.open(folder + filename + '.png'), dtype='float64')
        if len(clean.shape) > 2:
            clean = clean[:, :, 0]
        kernel = np.array(Image.open(folder + 'kernel.png', ), dtype='float64')[:, :, 0]

        skimage.io.imsave(folder + filename + '_blurred.png', make_blurred(clean, kernel, noise_level).astype('uint8'))
    else:
        # filename = 'avion'
        # noise_level = '5'
        # folder = 'my_data/'
        # sys.argv[1:] = [folder + filename + '_blurred.png',
        #                 folder + 'kernel.png',
        #                 folder + filename + '_output.png',
        #                 noise_level]
        parser = create_parser()
        namespace = parser.parse_args(sys.argv[1:])

        blurred = np.array(Image.open(namespace.blurred[0]), dtype='float64')
        if len(blurred.shape) > 2:
            blurred = blurred[:, :, 0]
        kernel = np.array(Image.open(namespace.kernel[0]), dtype='float64')[:, :, 0]
        kernel /= kernel.sum()
        noise_level = float(namespace.noise_level[0])

        clean = deblur(blurred, kernel, noise_level, iterations=100, alpha=noise_level / 5)
        skimage.io.imsave(namespace.output[0], clean.astype(np.uint8))
