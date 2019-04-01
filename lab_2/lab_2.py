import os
import cv2
import numpy as np
from math import exp, sqrt


def _def_operation(window, *args, **kwargs):
    return round(window.sum() / window.size)


def basic_filter(src, window_size, operation=_def_operation):
    rows, cols = img.shape[:2]

    out = np.empty(src.shape, src.dtype)

    step = (window_size - 1) // 2
    for i in range(rows):
        # in intervals everything that is less or equals to -rows == 0
        # and everything that is greater or equals to rows == rows
        r0 = i - step - rows
        r1 = i + step + 1
        for j in range(cols):
            c0 = j - step - cols
            c1 = j + step + 1
            window = src[r0:r1, c0:c1]
            out[i, j] = operation(window)

    return out


def median_operation(window):
    flat = sorted(window.flatten())
    center = (len(flat) - 1) // 2
    return flat[center]


def min_operation(window):
    return window.min()


def max_operation(window):
    return window.max()

def gaussian_kernel_1D(size, stdev):
    filter = []
    sqr = stdev ** 2
    i = (size - 1) // 2
    for x in range(-i, i + 1):
        tmp = 1 / sqrt(2 * np.pi * sqr)
        tmp = tmp * exp(- (x ** 2)/ (2 * sqr))
        filter.append(tmp)

    return np.array(filter)

def gauss_filter(src, size, stdev):
    ifilter = gaussian_kernel_1D(size, stdev)
    def operation(window):
        shape = window.shape
        if shape[1] < size: # * column
            out = window * ifilter[:shape[1]]
        else:
            out = window * ifilter
        if shape[0] < size:
            ifilter_T = ifilter[:shape[0],np.newaxis] 
        else:
            ifilter_T = ifilter[:,np.newaxis] 
        out *= ifilter_T # * row
        return out.sum()
    return basic_filter(src, size, operation)

def gen_weighted_filter(size):
    filter = np.zeros((size, size))
    weight = 2**(size-1)
    center = (size - 1) // 2
    _gen_weighted_filter(filter, weight, (center, center), size)
    return filter


def _gen_weighted_filter(filter, weight, idx, max_idx):
    if filter[idx]:
        return

    filter[idx] = weight
    if weight // 2 < 1:
        return

    r, c = idx
    weight //= 2
    if r - 1 >= 0:
        _gen_weighted_filter(filter, weight, (r - 1, c), max_idx)
    if r + 1 < max_idx:
        _gen_weighted_filter(filter, weight, (r + 1, c), max_idx)
    if c - 1 >= 0:
        _gen_weighted_filter(filter, weight, (r, c - 1), max_idx)
    if c + 1 < max_idx:
        _gen_weighted_filter(filter, weight, (r, c + 1), max_idx)


if __name__ == "__main__":
    img = cv2.imread('../imgs/2/lenaG.png', 0)

    os.makedirs('res', exist_ok=True)

    cv2.imwrite('res/filter_1.jpg', basic_filter(img, 3))

    cv2.imwrite('res/median.jpg', basic_filter(img, 9, median_operation))
    cv2.imwrite('res/min.jpg', basic_filter(img, 9, min_operation))
    cv2.imwrite('res/max.jpg', basic_filter(img, 9, max_operation))

    cv2.imwrite('res/gauss.jpg', gauss_filter(img, 9, 1))
