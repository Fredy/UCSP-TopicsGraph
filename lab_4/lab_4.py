import os
import cv2
import sys
import numpy as np


def rgb_to_cmy(src):
    # src is a np array
    return 255 - src


def cmy_to_rgb(src):
    return 255 - src


def rgb_to_cie(src):
    # No
    pass


def rgb_to_lab(src):
    # No
    pass


def rgb_to_hsv(src):
    # H 0, 360
    # S 0, 1
    # V 0, 1
    out = src / 255
    for i in range(src.shape[0]):  # rows
        for j in range(src.shape[1]):  # cols
            pixel = out[i, j]
            r, g, b = pixel
            min = pixel.min()
            max = pixel.max()

            # H
            if max == min:
                h = 0
            elif max == r:
                h = 60 * (g - b) / (max - min)
            elif max == g:
                h = 60 * (b - r) / (max - min) + 120
            else:
                h = 60 * (r - g) / (max - min) + 240

            if h < 0:
                h += 360

            # S
            if max != 0:
                s = (max - min) / max
            else:
                s = 0

            # V = max

            out[i, j] = [h, s, max]

    return out


def rgb_to_hsi(src):
    # H 0, 360
    # S 0, 1
    # L 0, 1
    out = src / 255
    for i in range(src.shape[0]):  # rows
        for j in range(src.shape[1]):  # cols
            pixel = out[i, j]
            r, g, b = pixel
            min = pixel.min()
            max = pixel.max()

            # H
            if max == min:
                h = 0
            elif max == r:
                h = 60 * (g - b) / (max - min)
            elif max == g:
                h = 60 * (b - r) / (max - min) + 120
            else:
                h = 60 * (r - g) / (max - min) + 240

            if h < 0:
                h += 360

            # S
            if max == 0 or max == 1:
                s = 0
            else:
                s = 1 - 3 / pixel.sum() * min
            # L

            l = pixel.mean()

            out[i, j] = [h, s, l]

    return out


def rgb_to_yuv(src):
    out = src / 255
    vals = np.array([
        [0.299, 0.587, 0.144],
        [-0.147, -0.289, 0.436],
        [0.615, -0.5151, -0.1]
    ])
    for i in range(src.shape[0]):  # rows
        for j in range(src.shape[1]):  # cols
            out[i, j] = vals @ out[i, j]

    return out


def rgb_to_yiq(src):
    out = src / 255
    vals = np.array([
        [0.299, 0.587, 0.144],
        [0.596, -0.275, -0.321],
        [0.212, -0.523, 0.311]
    ])
    for i in range(src.shape[0]):  # rows
        for j in range(src.shape[1]):  # cols
            out[i, j] = vals @ out[i, j]

    return out


def rgb_to_ycbcr(src):
    out = src / 255

    Kr = 0.299
    Kg = 0.587
    Kb = 0.144
    comp = np.array([16, 128, 128])
    weights = np.array([219, 244, 244])

    for i in range(src.shape[0]):  # rows
        for j in range(src.shape[1]):  # cols
            R, G, B = out[i, j]

            # Y = Kr * R + Kg * G + Kb * B
            # Pb = 0.5 * (B - Y) / (1 - Kb)
            # Pr = 0.5 * (R - Y) / (1 - Kr)

            # out[i,j] = comp + weights * [Y, Pb, Pr]
            Y = 16 + 65.481 * R + 128.553 * G + 24.966 * B
            Cb = 128 - 37.797 * R - 74.203 * G + 112 * B
            Cr = 128 + 112 * R - 93.786 * G - 18.214 * B

            out[i, j] = [Y, Cb, Cr]

    return out


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Syntax: file_name.py R G B")
        exit(-1)

    r = int(sys.argv[1])
    g = int(sys.argv[2])
    b = int(sys.argv[3])

    rgb = np.array([r, g, b], ndmin=3)

    print("CMY:", rgb_to_cmy(rgb)[0, 0])
    print("HSV:", rgb_to_hsv(rgb)[0, 0])
    print("HSI:", rgb_to_hsi(rgb)[0, 0])
    print("YUV:", rgb_to_yuv(rgb)[0, 0])
    print("YIQ:", rgb_to_yiq(rgb)[0, 0])
    print("YCbCr:", rgb_to_ycbcr(rgb)[0, 0])
