from PIL import Image
import numpy as np
import sys
import math
import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # mse (input_file_1) (input_file_2)
    gradient_parser = subparsers.add_parser("mse")
    gradient_parser.add_argument("params", nargs=2)

    # calc_noise (input_file)
    gradient_parser = subparsers.add_parser("calc_noise")
    gradient_parser.add_argument("params", nargs=1)

    # median (r) (input_file) (output_file)
    gradient_parser = subparsers.add_parser("median")
    gradient_parser.add_argument("params", nargs=3)

    # bilateral (sigma_d) (sigma_r) (input_file) (output_file)
    gradient_parser = subparsers.add_parser("bilateral")
    gradient_parser.add_argument("params", nargs=4)

    # query (noise_level)
    gradient_parser = subparsers.add_parser("query")
    gradient_parser.add_argument("params", nargs=1)

    # denoise (input_file) (output_file)
    gradient_parser = subparsers.add_parser("denoise")
    gradient_parser.add_argument("params", nargs=2)

    return parser


def normalize(image):
    image[image < 0] = 0
    image[image > 255] = 255
    return image.astype('uint8')


def make_image(array):
    return Image.fromarray(normalize(array))


def gen_noisy_image(image, sigma):
    clean = np.array(Image.open(image), dtype="float64")
    noise = np.random.normal(0, sigma, clean.shape[:2])
    clean[..., 0] += noise
    clean[..., 1] += noise
    clean[..., 2] += noise
    return make_image(clean)


# mse (input_file_1) (input_file_2)
def mse(params):
    image1 = np.array(Image.open(params[0]), dtype="float64")
    image2 = np.array(Image.open(params[1]), dtype="float64")
    return math.sqrt(((image1 - image2) ** 2).sum() / image1.size)


# calc_noise (input_file)
def calc_noise(params):
    pass


# median (r) (input_file) (output_file)
def median(params):
    pass


# bilateral (sigma_d) (sigma_r) (input_file) (output_file)
def bilateral(params):
    pass


# query (noise_level)
def query(params):
    pass


# denoise (input_file) (output_file)
def denoise(params):
    pass


# for testing
def get_difference(first, second, out):
    image1 = np.array(Image.open(first), dtype="float64")[:, :, :3]
    image2 = np.array(Image.open(second), dtype="float64")[:, :, :3]

    result = np.vectorize(abs)(image1 - image2)
    make_image(result * 4).save(out)


if __name__ == '__main__':
    # for testing
    generating = 1
    if not generating:
        sys.argv[1:] = ["mse", "noisy.bmp", "../img/lena.bmp"]

        parser = create_parser()
        namespace = parser.parse_args(sys.argv[1:])

        command = globals()[namespace.command]
        result = command(namespace.params)
        if str(type(result)) == "<class 'PIL.Image.Image'>":
            result.save(namespace.params[-1])
        else:
            print(result)
    else:
        source = "../img/lena.bmp"
        noisy = "noisy.bmp"
        result = gen_noisy_image(source, 32)
        result.save(noisy)
        get_difference(source, noisy, "res.bmp")
