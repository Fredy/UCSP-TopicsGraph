import cv2
import numpy as np
from math import (cos, sin, radians, floor, ceil, exp, log)


def translation(src, x, y):
    out = np.zeros(src.shape, src.dtype)
    X, Y, _ = src.shape
    out[-X + x: X + x, -Y + y: Y + y] = src[-X - x: X - x, -Y - y: Y - y]
    return out


def resize(src, xfactor, yfactor):
    rows, cols, c = src.shape
    shape = (floor(rows * yfactor), floor(cols * xfactor), c)
    out = np.empty(shape, src.dtype)

    for i in range(shape[0]):
        for j in range(shape[1]):
            out[i,j] = src[floor(i // yfactor), floor(j // xfactor)]

    return out


def rotate(src, angle, pivot=None):
    """
    :params pivot: Center of rotation.    
    """
    shape = src.shape
    out = np.zeros(shape, src.dtype)

    if pivot == None:
        pivot = (shape[1] // 2, shape[0] // 2)
    print(pivot)

    for r in range(shape[0]):
        for c in range(shape[1]):
            rc, rr = _rotate_point((c, r), pivot, angle)
            rc = round(rc)
            rr = round(rr)
            if (rr >= shape[0] or rr < 0 or
                    rc >= shape[1] or rc < 0):
                continue
            out[rr, rc] = src[r, c]

    return out


def _rotate_point(point, center, angle):
    """Angle in degrees"""
    x, y = point
    cx, cy = center
    angle = radians(angle)
    _cos = cos(angle)
    _sin = sin(angle)

    x = (x - cx) * _cos - (y - cy) * _sin + cx
    y = (x - cx) * _sin + (y - cy) * _cos + cy

    return (abs(x), abs(y))

@np.vectorize
def exp_filter(x):
    return pow(2, exp) % 255

@np.vectorize
def log_filter(x):
    if x == 0:
        return 0
    return log(x)

@np.vectorize
def inverse_filter(x):
    return 255 - x

if __name__ == "__main__":
    img = cv2.imread('../imgs/woman.jpg')
    positive = translation(img, 50, 40)
    negative = translation(img, -50, -40)
    cv2.imwrite('translation_positive.jpg', positive)
    cv2.imwrite('translation_negative.jpg', negative)

    cv2.imwrite('rotate.jpg', rotate(img, 90))

    cv2.imwrite('upscale.jpg', resize(img, 3,3))
    cv2.imwrite('downscale.jpg', resize(img, 0.4,0.4))

    cv2.imwrite('exp.jpg', exp_filter(img))
    cv2.imwrite('log.jpg', log_filter(img))
    cv2.imwrite('inverse.jpg', inverse_filter(img))