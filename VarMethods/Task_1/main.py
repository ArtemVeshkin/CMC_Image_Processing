from PIL import Image
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.filters import sobel, gaussian
from scipy.interpolate import RectBivariateSpline, interp1d
from collections import deque


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', nargs=1)
    parser.add_argument('initial_snake', nargs=1)
    parser.add_argument('output_image', nargs=1)
    parser.add_argument('alpha', nargs=1)
    parser.add_argument('beta', nargs=1)
    parser.add_argument('tau', nargs=1)
    parser.add_argument('w_line', nargs=1)
    parser.add_argument('w_edge', nargs=1)
    parser.add_argument('kappa', nargs=1)

    return parser


def display_image_in_actual_size(img, dpi=80):
    height = img.shape[0]
    width = img.shape[1]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(img, cmap='gray')

    # plt.show()

    return fig, ax


def save_mask(fname, snake, img):
    plt.ioff()
    fig, ax = display_image_in_actual_size(img)
    ax.fill(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    fig.savefig(fname, pad_inches=0, bbox_inches='tight', dpi='figure')
    plt.close(fig)

    mask = skio.imread(fname)
    blue = ((mask[:, :, 2] == 255) & (mask[:, :, 1] < 255) & (mask[:, :, 0] < 255)) * 255
    blue = np.array(blue, dtype='uint8')
    skio.imsave(fname, blue)
    plt.ion()


def display_snake(img, init_snake, result_snake, fname="default"):
    fig, ax = display_image_in_actual_size(img)
    ax.plot(init_snake[:, 0], init_snake[:, 1], '-r', lw=2)
    ax.plot(result_snake[:, 0], result_snake[:, 1], '-b', lw=2)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    # plt.show()
    if fname != "default":
        fig.savefig(fname, pad_inches=0, bbox_inches='tight', dpi='figure')
    plt.close()


def reparametrization(snake):
    n = snake.shape[0]
    snake_len = np.zeros(n)
    total_len = 0
    for i in range(n):
        total_len += np.linalg.norm(snake[i] - snake[i - 1])
        snake_len[i] = total_len

    x = interp1d(snake_len, snake[:, 0])
    y = interp1d(snake_len, snake[:, 1])

    for i in range(0, snake.shape[0] - 1):
        snake[i][0] = x(i * total_len / snake.shape[0])
        snake[i][1] = y(i * total_len / snake.shape[0])

    return snake


def active_contour(image, snake, alpha=1., beta=1., w_line=0, w_edge=1, tau=0.025, kappa=0.01):
    source = image
    image = gaussian(image, 3)
    edge = sobel(image)
    image = w_line * image + w_edge * edge

    interpolated_image = RectBivariateSpline(np.arange(image.shape[1]), np.arange(image.shape[0]), image.T, kx=2, ky=2,
                                             s=0)

    # Create matrix for Euler equation
    n = snake.shape[0]
    a = np.roll(np.eye(n), -1, axis=0) + np.roll(np.eye(n), -1, axis=1) - 2 * np.eye(n)
    b = np.roll(np.eye(n), -2, axis=0) + np.roll(np.eye(n), -2, axis=1) - 4 * np.roll(np.eye(n), -1, axis=0) - \
        4 * np.roll(np.eye(n), -1, axis=1) + 6 * np.eye(n)
    A = -alpha * a + beta * b
    inv_A = np.linalg.inv(A + tau * np.eye(n))

    x, y = snake[:, 0], snake[:, 1]
    x_normal = np.zeros(n)
    y_normal = np.zeros(n)

    prev_snake = deque()

    for i in range(300):
        # For testing
        # if i % 5 == 0:
        #     display_snake(source, np.stack([x, y], axis=1), np.stack([x, y], axis=1),
        #                   "Iterations/iteration_" + str(i) + ".png")
        fx = interpolated_image(x, y, dx=1, grid=False)
        fy = interpolated_image(x, y, dy=1, grid=False)
        fx /= np.linalg.norm(fx)
        fy /= np.linalg.norm(fy)

        for j in range(n - 1):
            x_normal[j] = y[j + 1] - y[j]
            y_normal[j] = x[j] - x[j + 1]

        xn = inv_A @ (tau * x + fx + kappa * x_normal)
        yn = inv_A @ (tau * y + fy + kappa * y_normal)

        dx = xn - x
        dy = yn - y
        x += dx
        y += dy
        x[-1] = x[0]
        y[-1] = y[0]

        snake = reparametrization(np.stack([y, x], axis=1))
        snake[snake < 0] = 0
        snake[snake[:, 0] > image.shape[1] - 1, 0] = image.shape[1] - 1
        snake[snake[:, 1] > image.shape[0] - 1, 1] = image.shape[0] - 1
        x = snake[:, 1]
        y = snake[:, 0]

        if len(prev_snake) >= 20:
            prev_snake.popleft()

        min_dist = n * 100
        for pair in prev_snake:
            cur_dist = np.average(np.abs(pair[0] - x) + np.abs(pair[1] - y))
            if cur_dist < min_dist:
                min_dist = cur_dist
        prev_snake.append([x.copy(), y.copy()])

        if min_dist < 0.1:
            print("Stopped on " + str(i) + " iteration")
            break
    return np.stack([x, y], axis=1)


if __name__ == '__main__':
    # For testing
    # testdata = "astranaut"
    # sys.argv[1:] = ["task2_testdata/" + testdata + ".png", "task2_testdata/" + testdata + "_init_snake.txt", "mask.png",
    #                 "0.7",  "1.0", "0.025", "0.1", "1.9", "0.0001"]

    parser = create_parser()
    namespace = parser.parse_args(sys.argv[1:])

    input_image = namespace.input_image[0]
    initial_snake = namespace.initial_snake[0]
    output_image = namespace.output_image[0]
    alpha = float(namespace.alpha[0])
    beta = float(namespace.beta[0])
    tau = float(namespace.tau[0])
    w_line = float(namespace.w_line[0])
    w_edge = float(namespace.w_edge[0])
    kappa = float(namespace.kappa[0])

    img = np.array(Image.open(input_image), dtype="float64")
    snake = active_contour(img, np.loadtxt(initial_snake), alpha=alpha, beta=beta, kappa=kappa, w_line=w_line,
                           w_edge=w_edge, tau=tau)
    save_mask(output_image, snake, img)
