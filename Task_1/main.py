from PIL import Image, ImageDraw
import numpy as np
import sys
import argparse
import math


def create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # mirror {x|y}
    mirror_parser = subparsers.add_parser("mirror")
    mirror_parser.add_argument("params", nargs=1)

    # extract (left_x) (top_y) (width) (height)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("params", nargs=4)

    # rotate {cw|ccw} (angle)
    rotate_parser = subparsers.add_parser("rotate")
    rotate_parser.add_argument("params", nargs=2)

    # extend {dup|even|odd} (size)
    extend_parser = subparsers.add_parser("extend")
    extend_parser.add_argument("params", nargs=2)

    # gauss (sigma)
    gauss_parser = subparsers.add_parser("gauss")
    gauss_parser.add_argument("params", nargs=1)

    # gauss (sigma)
    gradient_parser = subparsers.add_parser("gradient")
    gradient_parser.add_argument("params", nargs=1)

    parser.add_argument("input_file", nargs=1)
    parser.add_argument("output_file", nargs=1)

    return parser


def normalize(image):
    image[image < 0] = 0
    image[image > 255] = 255
    return image.astype('uint8')


# mirror {x|y}
def mirror(params, input_file):
    image = np.array(Image.open(input_file))

    if params[0] == "x":
        image = image[:, ::-1, :]
    elif params[0] == "y":
        image = image[::-1, :, :]
    else:
        sys.exit(1)

    return Image.fromarray(image)


# extract (left_x) (top_y) (width) (height)
def extract(params, input_file):
    params = list(map(int, params))

    image = Image.open(input_file)
    pixels = image.load()

    new_image = Image.new("RGBA", (params[2], params[3]))
    new_image_draw = ImageDraw.Draw(new_image)

    for i in range(0, params[2]):
        for j in range(0, params[3]):
            new_image_draw.point((i, j), pixels[i + params[0], j + params[1]])

    del new_image_draw
    return new_image


# rotate {cw|ccw} (angle)
def rotate(params, input_file):
    params[1] = int(params[1])

    image = np.array(Image.open(input_file))

    times_to_rotate = (params[1] // 90) % 4
    if params[0] == "cww":
        times_to_rotate = 4 - times_to_rotate
    elif params[0] != "cw":
        sys.exit(1)

    for i in range(times_to_rotate):
        image = image.transpose((1, 0, 2))[:, ::-1, :]

    return Image.fromarray(image)


def dup(image, size):
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

    return Image.fromarray(new_image_pixels)


def next_pos(cur, size, direction):
    if direction == 0:
        cur -= 1
        if cur <= 0:
            direction = 1
            cur = 0
    else:
        cur += 1
        if cur >= size - 1:
            direction = 0
            cur = size - 1

    return cur, direction


def even(image, size):
    pixels = np.array(image)[:, :, :3]
    width, height = image.size
    new_pixels = np.zeros((height + 2 * size, width + 2 * size, 3), dtype="uint8")

    new_pixels[size: -size, size: -size] += pixels

    cur1, cur2, direction1, direction2 = 0, 0, 1, 0
    for step in range(size):
        cur1, direction1 = next_pos(cur1, width, direction1)
        cur2, direction2 = next_pos(cur2, width, direction2)
        for i in range(height):
            new_pixels[i + size, size - step - 1] = new_pixels[i + size, size + cur1]
            new_pixels[i + size, size + width + step] = new_pixels[i + size, size + width - cur2 - 2]

    cur1, cur2, direction1, direction2 = 0, 0, 1, 0
    for step in range(size):
        cur1, direction1 = next_pos(cur1, height, direction1)
        cur2, direction2 = next_pos(cur2, height, direction2)
        for i in range(width + size * 2):
            new_pixels[size - step - 1, i] = new_pixels[size + cur1, i]
            new_pixels[size + height + step, i] = new_pixels[size + height - cur2 - 2, i]

    return Image.fromarray(new_pixels)


def odd(image, size):
    pixels = np.array(image)[:, :, :3]
    width, height = image.size
    new_pixels = np.zeros((height + 2 * size, width + 2 * size, 3))

    new_pixels[size: -size, size: -size] += pixels

    cur1, cur2, direction1, direction2 = 0, 0, 1, 0
    for step in range(size):
        cur1, direction1 = next_pos(cur1, width, direction1)
        cur2, direction2 = next_pos(cur2, width, direction2)
        for i in range(height):
            new_pixels[i + size, size - step - 1] = 2 * new_pixels[i + size, size] \
                                                    - new_pixels[i + size, size + cur1]
            new_pixels[i + size, size + width + step] = 2 * new_pixels[i + size, size + width - 1] \
                                                        - new_pixels[i + size, size + width - cur2 - 2]

    cur1, cur2, direction1, direction2 = 0, 0, 1, 0
    for step in range(size):
        cur1, direction1 = next_pos(cur1, height, direction1)
        cur2, direction2 = next_pos(cur2, height, direction2)
        for i in range(width + size * 2):
            new_pixels[size - step - 1, i] = 2 * new_pixels[size, i] \
                                             - new_pixels[size + cur1, i]
            new_pixels[size + height + step, i] = 2 * new_pixels[size + height - 1, i] \
                                                  - new_pixels[size + height - cur2 - 2, i]

    return Image.fromarray(normalize(new_pixels))


# extend {dup|even|odd} (size)
def extend(params, input_file):
    params[1] = int(params[1])

    image = Image.open(input_file)
    if params[0] == "dup":
        new_image = dup(image, params[1])
        del image
        image = new_image
    elif params[0] == "even":
        new_image = even(image, params[1])
        del image
        image = new_image
    elif params[0] == "odd":
        new_image = odd(image, params[1])
        del image
        image = new_image
    else:
        sys.exit(1)

    return image


def generate_mask(ms, func):
    mask = np.zeros((ms * 2 + 1, ms * 2 + 1))
    for i in range(-ms, ms + 1):
        for j in range(-ms, ms + 1):
            mask[ms + i, ms + j] = func(i, j)
    return mask


# gauss (sigma)
def gauss(params, input_file):
    sigma = float(params[0])
    # Mask size / 2
    ms = math.ceil(sigma * 3)

    # Using 6*sigma x 6*sigma Gaussian mask
    mask = generate_mask(ms, lambda x, y: math.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2))))
    mask /= mask.sum()

    pixels = np.array(extend(["dup", str(ms)], input_file))
    new_pixels = np.zeros((pixels.shape[0] - ms * 2, pixels.shape[1] - ms * 2, 3))

    for i in range(new_pixels.shape[0]):
        for j in range(new_pixels.shape[1]):
            # Convolution
            for c in range(3):
                new_pixels[i, j, c] = pixels[i:i + 2 * ms + 1, j:j + 2 * ms + 1, c].dot(mask).diagonal().sum()

    return Image.fromarray(normalize(new_pixels))


# gradient (sigma)
def gradient(params, input_file):
    sigma = float(params[0])

    # Mask size / 2
    ms = math.ceil(sigma * 3.5)
    x_mask = generate_mask(ms, lambda x, y: -x / (sigma ** 2) * math.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2))))
    y_mask = generate_mask(ms, lambda x, y: -y / (sigma ** 2) * math.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2))))

    pixels = np.array(extend(["dup", str(ms)], input_file))

    # Converting to grayscale
    pixels[:, :, 0] = 0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]

    new_pixels = np.zeros((pixels.shape[0] - ms * 2, pixels.shape[1] - ms * 2, 3))
    # Calculating gradient
    for i in range(new_pixels.shape[0]):
        for j in range(new_pixels.shape[1]):
            # Convolution
            horizontal = pixels[i:i + 2 * ms + 1, j:j + 2 * ms + 1, 0].dot(x_mask).diagonal().sum()
            vertical = pixels[i:i + 2 * ms + 1, j:j + 2 * ms + 1, 0].dot(y_mask).diagonal().sum()

            new_pixels[i, j, 0] \
                = new_pixels[i, j, 1] \
                = new_pixels[i, j, 2] = math.sqrt(horizontal ** 2 + vertical ** 2)

    new_pixels *= 255 / new_pixels.max(initial=255)
    return Image.fromarray(normalize(new_pixels))


# for testing
def get_difference(my_file, right_file, out):
    my_image = np.array(Image.open(my_file))[:, :, :3]
    right_image = np.array(Image.open(right_file))[:, :, :3]

    result = np.zeros((max(my_image.shape[0], right_image.shape[0]), max(my_image.shape[1], right_image.shape[1]), 3))
    result[0:my_image.shape[0], 0:my_image.shape[1], :] += my_image
    result[0:right_image.shape[0], 0:right_image.shape[1], :] = abs(result[0:right_image.shape[0], 0:right_image.shape[1], :] - right_image)
    Image.fromarray(normalize(result) * 10).save(out)


if __name__ == '__main__':
    # for testing
    # sys.argv[1:] = ["extend", "odd", "7", "testing/parrot.png", "res.bmp"]

    parser = create_parser()
    namespace = parser.parse_args(sys.argv[1:])

    command = globals()[namespace.command]
    changed_image = command(namespace.params, namespace.input_file[0])
    changed_image.save(namespace.output_file[0])

    # for testing
    # get_difference("res.bmp", "testing/14.png", "difference.bmp")
