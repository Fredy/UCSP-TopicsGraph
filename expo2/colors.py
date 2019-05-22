import os
import cv2
import sys
from math import acos, sqrt, degrees
import numpy as np


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


def rgb_to_yuv(src):
    out = src / 255
    vals = np.array([
        [0.299,  0.587, 0.144],
        [-0.147, -0.289, 0.436],
        [0.615, -0.5151, -0.1]
    ])
    for i in range(src.shape[0]):  # rows
        for j in range(src.shape[1]):  # cols
            out[i, j] = vals @ out[i, j]

    return out

if __name__ == "__main__":
    img_name = '../imgs/2/lenaS.jpg'
    if len(sys.argv) > 1:
        img_name = sys.argv[1]
    img = cv2.imread(img_name, 0)

    if img is None:
        print('Can not read `{}`'.format(img_name))
        exit(-1)

    os.makedirs('res', exist_ok=True)

 # ...
