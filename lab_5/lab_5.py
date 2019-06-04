import os
import cv2
import sys
from multiprocessing import Pool
import numpy as np


def normalize(img):
    vmin = img.min()
    vmax = img.max()

    img = (img - vmin) / vmax * 255


def _border3(src, kernel, t=None):
    out = np.zeros(src.shape, src.dtype)

    rows, cols = img.shape[:2]

    for i in range(1, rows-1):
        r0 = i - 1
        r1 = i + 2
        for j in range(1, cols - 1):
            c0 = j - 1
            c1 = j + 2
            window = src[r0:r1, c0: c1]
            res = abs((window * kernel).sum()) 
            if t:
                out[i, j] = 255 if res > t else 0
            else:
                out[i, j] = min(255, res)

    return out


def point_detection(src):
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
    return _border3(src, kernel, 100)


def horizontal(src):
    kernel = np.array([
        [-1, -1, -1],
        [2, 2, 2],
        [-1, -1, -1]
    ])
    return _border3(src, kernel)


def positive_45(src):
    kernel = np.array([
        [-1, -1, 2],
        [-1, 2, -1],
        [2, -1, -1]
    ])
    return _border3(src, kernel)


def vertical(src):
    kernel = np.array([
        [-1, 2, -1],
        [-1, 2, -1],
        [-1, 2, -1]
    ])
    return _border3(src, kernel)


def negative_45(src):
    kernel = np.array([
        [2, -1, -1],
        [-1, 2, -1],
        [-1, -1, 2]
    ])
    return _border3(src, kernel)


if __name__ == "__main__":
    img_name = '../imgs/2/lenaS.jpg'
    if len(sys.argv) > 1:
        img_name = sys.argv[1]
    img = cv2.imread(img_name, 0)

    if img is None:
        print('Can not read `{}`'.format(img_name))
        exit(-1)

    os.makedirs('res', exist_ok=True)

    # cv2.imwrite('res/point_detection.jpg', point_detection(img))
    # cv2.imwrite('res/horizontal.jpg', horizontal(img))
    # cv2.imwrite('res/vertical.jpg', vertical(img))
    # cv2.imwrite('res/positivie_45.jpg', positive_45(img))
    # cv2.imwrite('res/negative_45.jpg', negative_45(img))

    with Pool() as pool:
        res_point = pool.apply_async(
            point_detection, (img,),
            callback=lambda x: cv2.imwrite('res/point_detection.jpg', x)
        )
        res_horizontal = pool.apply_async(
            horizontal, (img,),
            callback=lambda x: cv2.imwrite('res/horizontal.jpg', x)
        )
        res_vertical = pool.apply_async(
            vertical, (img,),
            callback=lambda x: cv2.imwrite('res/vertical.jpg', x)
        )

        res_positive_45 = pool.apply_async(
            positive_45, (img,),
            callback=lambda x: cv2.imwrite('res/positive_45.jpg', x)
        )
        res_negative_45 = pool.apply_async(
            negative_45, (img,),
            callback=lambda x: cv2.imwrite('res/negative_45.jpg', x)
        )

        res_point.get()
        res_horizontal.get()
        res_vertical.get()
        res_positive_45.get()
        res_negative_45.get()
