from PIL import Image
import numpy as np
import sys
import math
import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # mse (input_file_1) (input_file_2)
    subparser = subparsers.add_parser("mse")
    subparser.add_argument("params", nargs=2)

    # calc_noise (input_file)
    subparser = subparsers.add_parser("calc_noise")
    subparser.add_argument("params", nargs=1)

    # median (r) (input_file) (output_file)
    subparser = subparsers.add_parser("median")
    subparser.add_argument("params", nargs=3)

    # bilateral (sigma_d) (sigma_r) (input_file) (output_file)
    subparser = subparsers.add_parser("bilateral")
    subparser.add_argument("params", nargs=4)

    # query (noise_level)
    subparser = subparsers.add_parser("query")
    subparser.add_argument("params", nargs=1)

    # denoise (input_file) (output_file)
    subparser = subparsers.add_parser("denoise")
    subparser.add_argument("params", nargs=2)

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


def extend(image, size):
    pixels = np.array(image)[:, :, :3]
    width, height = image.size
    new_image_pixels = np.zeros((height + 2 * size, width + 2 * size, 3), dtype=np.uint8)

    for i in range(new_image_pixels.shape[0]):
        for j in range(new_image_pixels.shape[1]):
            if (size <= i < height + size) and (size <= j < width + size):
                new_image_pixels[i, j] = pixels[i - size, j - size]
            elif i < size:
                if j < size:
                    new_image_pixels[i, j] = pixels[0, 0]
                elif j < size + width:
                    new_image_pixels[i, j] = pixels[0, j - size]
                else:
                    new_image_pixels[i, j] = pixels[0, width - 1]
            elif size <= i < height + size:
                if j < size:
                    new_image_pixels[i, j] = pixels[i - size, 0]
                else:
                    new_image_pixels[i, j] = pixels[i - size, width - 1]
            else:
                if j < size:
                    new_image_pixels[i, j] = pixels[height - 1, 0]
                elif j < size + width:
                    new_image_pixels[i, j] = pixels[height - 1, j - size]
                else:
                    new_image_pixels[i, j] = pixels[height - 1, width - 1]

    return new_image_pixels


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
    r = int(params[0])
    noisy = np.array(Image.open(params[1]))
    clean = noisy
    height, width = noisy.shape[0:2]
    for i in range(height):
        print("progress:" + str(i) + "/" + str(height))
        for j in range(width):
            clean[i, j, 0] = np.median(
                noisy[max(0, i - r):min(height, i + r + 1), max(0, j - r):min(width, j + r + 1), 0])
            clean[i, j, 1] = np.median(
                noisy[max(0, i - r):min(height, i + r + 1), max(0, j - r):min(width, j + r + 1), 1])
            clean[i, j, 2] = np.median(
                noisy[max(0, i - r):min(height, i + r + 1), max(0, j - r):min(width, j + r + 1), 2])

    return make_image(clean)


def gauss(x, sigma):
    return 1 / (2 * np.pi * sigma ** 2) * np.exp(-x ** 2 / (2 * sigma ** 2))


def generate_mask(mask_size, sigma_d, sigma_r, intensities):
    mask = np.zeros((mask_size * 2 + 1, mask_size * 2 + 1))
    for i in range(-mask_size, mask_size + 1):
        for j in range(-mask_size, mask_size + 1):
            mask[mask_size + i, mask_size + j] = gauss(np.sqrt(i ** 2 + j ** 2), sigma_d) \
                                                 * gauss(intensities[i, j], sigma_r)
    return mask


# bilateral (sigma_d) (sigma_r) (input_file) (output_file)
def bilateral(params):
    sigma_d = float(params[0])
    sigma_r = float(params[1])

    mask_size = 2
    image = extend(Image.open(params[2]), mask_size) if str(type(params[2])) != "<class 'PIL.Image.Image'>" \
        else extend(params[2], mask_size)
    new_image = np.array(Image.open(params[2])) if str(type(params[2])) != "<class 'PIL.Image.Image'>" \
        else np.array(params[2])

    for i in range(new_image.shape[0]):
        print("progress:" + str(i) + "/" + str(new_image.shape[0]))
        for j in range(new_image.shape[1]):
            intensity = image[i:i + 2 * mask_size + 1, j:j + 2 * mask_size + 1] - image[i + mask_size, j + mask_size]
            intensity = (intensity[:, :, 0] + intensity[:, :, 1] + intensity[:, :, 2]) / 3
            mask = generate_mask(mask_size, sigma_d, sigma_r, np.abs(intensity))
            mask /= mask.sum()
            for c in range(3):
                new_image[i, j, c] = image[i:i + 2 * mask_size + 1, j:j + 2 * mask_size + 1, c] \
                    .dot(mask).diagonal().sum()

    return Image.fromarray(normalize(new_image))


# query (noise_level)
def query(params):
    pass


# denoise (input_file) (output_file)
def denoise(params):
    r = 1
    sigma_d = 2
    sigma_r = 16
    image = median([r, params[0], params[1]])
    return bilateral([sigma_d, sigma_r, image, params[1]])



# for testing
def get_difference(first, second):
    image1 = np.array(Image.open(first), dtype="float64")[:, :, :3]
    image2 = np.array(Image.open(second), dtype="float64")[:, :, :3]

    result = np.vectorize(abs)(image1 - image2)
    return make_image(result * 4)


if __name__ == '__main__':
    # for testing
    generating = 0
    if not generating:
        # sys.argv[1:] = ["median", "1", "noisy.bmp", "res.bmp"]
        # sys.argv[1:] = ["bilateral", "2", "64", "noisy.bmp", "res.bmp"]
        # sys.argv[1:] = ["mse", "../img/lena.bmp", "res.bmp"]
        # sys.argv[1:] = ["denoise", "noisy.bmp", "res.bmp"]

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
        result = gen_noisy_image(source, 16)
        result.save(noisy)
        get_difference(source, noisy).save("res.bmp")
