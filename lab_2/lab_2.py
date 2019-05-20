import os
import cv2
import numpy as np
from math import exp, sqrt
import sys

from charts import draw_comp_hist


def _def_operation(window, *args, **kwargs):
    return round(window.sum() / window.size)


def basic_filter(src, window_size, operation=_def_operation):
    rows, cols = src.shape[:2]

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

    if stdev <= 0:
        stdev = ((size-1)*0.5 - 1)*0.3 + 0.8

    sqr = 2 * stdev ** 2
    i = (size - 1) // 2
    sum = 0
    for x in range(-i, i + 1):
        tmp = exp(- (x ** 2) / sqr)
        sum += tmp
        filter.append(tmp)

    sum = 1/sum
    return np.array(filter) * sum


def gauss_filter(src, window_size, stdev):
    out = np.empty(src.shape, src.dtype)

    rows, cols = src.shape[:2]
    step = (window_size - 1) // 2
    filter1D = gaussian_kernel_1D(window_size, stdev)

    for i in range(rows):
        # in intervals everything that is less or equals to -rows == 0
        # and everything that is greater or equals to rows == rows
        r0 = i - step - rows
        r1 = i + step + 1

        rleft = 0
        rright = rows
        if -rows > r0:
            rleft = - rows - r0
        if rows < r1:
            rright = rows - r1

        vfilter = filter1D[rleft: rright, np.newaxis]

        for j in range(cols):
            c0 = j - step - cols
            c1 = j + step + 1

            cleft = 0
            cright = cols
            if -cols > c0:
                cleft = - cols - c0
            if cols < c1:
                cright = cols - c1

            hfilter = filter1D[cleft: cright]

            window = src[r0:r1, c0:c1]

            tmp = window * vfilter * hfilter
            out[i, j] = tmp.sum()

    return out


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


def histogram_expansion(img, min=0, max=255):
    min_value = img.min()
    max_value = img.max()

    tmp = (max - min) / (max_value - min_value)
    out = (img - min_value) * tmp + min

    return out


def histogram_equalization(img, colors=256):
    out = np.empty(img.shape, img.dtype)

    hist = np.histogram(img, bins=np.arange(colors + 1), density=True)[0]
    hist = hist.cumsum() * (colors - 1)
    new_values = [round(i) for i in hist]
    print(len(new_values))

    rows, cols = out.shape[:2]
    for i in range(rows):
        for j in range(cols):
            # print(img[i,j])
            out[i,j] = new_values[img[i,j]]

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

    # cv2.imwrite('res/filter_1.jpg', basic_filter(img, 9))

    # cv2.imwrite('res/median.jpg', basic_filter(img, 9, median_operation))
    # cv2.imwrite('res/min.jpg', basic_filter(img, 9, min_operation))
    # cv2.imwrite('res/max.jpg', basic_filter(img, 9, max_operation))

    # gauss_res = gauss_filter(img, 5, 0)
    # cv2.imwrite('res/gauss.jpg', np.append(img, gauss_res, axis=1))

    expansion = histogram_expansion(img)
    cv2.imwrite('res/hist_expansion.jpg', expansion)

    draw_comp_hist(img, expansion, 'res/hist_expansion_chart.svg')

    equalization = histogram_equalization(img)
    cv2.imwrite('res/hist_equalization.jpg', equalization)
    draw_comp_hist(img, equalization, 'res/hist_equalization_chart.svg')