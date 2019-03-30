import os
import cv2
import numpy as np
from math import (cos, sin, radians, floor, ceil, exp, log)


def translation(src, x, y):
    out = np.zeros(src.shape, src.dtype)
    X, Y = src.shape[:2]
    out[-X + x: X + x, -Y + y: Y + y] = src[-X - x: X - x, -Y - y: Y - y]
    return out


def resize(src, xfactor, yfactor):
    rows, cols = src.shape[:2]
    if len(shape) == 2:
        shape = (floor(rows * yfactor), floor(cols * xfactor))
    else:
        shape = (floor(rows * yfactor), floor(cols * xfactor), shape[2])

    out = np.empty(shape, src.dtype)

    for i in range(shape[0]):
        for j in range(shape[1]):
            out[i, j] = src[floor(i // yfactor), floor(j // xfactor)]

    return out


def rotate(src, angle, pivot=None):
    """
    :params pivot: Center of rotation.    
    """
    shape = src.shape
    out = np.zeros(shape, src.dtype)

    if pivot is None:
        pivot = (shape[0] // 2, shape[1] // 2)

    for i in range(shape[0]):  # rows
        for j in range(shape[1]):  # cols
            x, y = _rotate_pos((shape[1] - j - 1 , i), pivot, angle)
            x = shape[1] - x - 1

            x = round(x)
            y = round(y)

            if 0 <= x < shape[1] and 0 <= y < shape[1]:
                out[y,x] = src[i, j]
        
    return out


def _rotate_pos(point, center, angle):
    """:params angle: in degrees"""
    angle = radians(angle)
    x, y = point
    cx, cy = center
    _cos = cos(angle)
    _sin = sin(angle)

    x -= cx
    y -= cy

    nx = x * _cos - y * _sin
    ny = x * _sin + y * _cos

    return nx + cx, ny + cy


def exp_filter(x):
    return pow(x, 2.5) % 255

def exp_filter_B(x):
    return pow(x / 255, 2.5) * 255

@np.vectorize
def log_filter(x):
    if x == 0:
        return 0
    x /= 255
    return (log(x) * 255) % 255

@np.vectorize
def log_filter_B(x):
    if x == 0:
        return 0
    x /= 255
    return 0 if log(x) < log(0.5) else 255

@np.vectorize
def inverse_filter(x):
    return 255 - x


if __name__ == "__main__":
    img = cv2.imread('../imgs/woman.jpg', 0)

    os.makedirs('res', exist_ok=True)

    positive = translation(img, 50, 40)
    negative = translation(img, -50, -40)
    cv2.imwrite('res/translation_positive.jpg', positive)
    cv2.imwrite('res/translation_negative.jpg', negative)

    cv2.imwrite('res/rotate.jpg', rotate(img, 15))

    cv2.imwrite('res/upscale.jpg', resize(img, 3, 3))
    cv2.imwrite('res/downscale.jpg', resize(img, 0.4, 0.4))

    cv2.imwrite('res/exp.jpg', exp_filter(img))
    cv2.imwrite('res/expB.jpg', exp_filter_B(img))
    cv2.imwrite('res/log.jpg', log_filter(img))
    cv2.imwrite('res/logB.jpg', log_filter_B(img))
    cv2.imwrite('res/inverse.jpg', inverse_filter(img))
