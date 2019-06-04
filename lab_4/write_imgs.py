import sys
import os
import cv2
import lab_4
import numpy as np


def write_rgb(img, file_name):
    r = np.zeros(img.shape, img.dtype)
    g = np.zeros(img.shape, img.dtype)
    b = np.zeros(img.shape, img.dtype)

    r[:, :, 2] = img[:, :, 2]  # 0, 255
    g[:, :, 1] = img[:, :, 1]  # 0, 255
    b[:, :, 0] = img[:, :, 0]  # 0, 255

    tmp = np.concatenate((r, g, b, img), axis=1)
    cv2.imwrite(file_name + '_rgb.jpg', tmp)


def write_cmy(img, file_name):
    c = np.zeros(img.shape, img.dtype)
    m = np.zeros(img.shape, img.dtype)
    y = np.zeros(img.shape, img.dtype)

    c[:, :, 2] = img[:, :, 2]  # 0, 255
    m[:, :, 1] = img[:, :, 1]  # 0, 255
    y[:, :, 0] = img[:, :, 0]  # 0, 255

    tmp = np.concatenate((c, m, y, img), axis=1)
    cv2.imwrite(file_name + '_cmy.jpg', tmp)


def write_hsv(img, file_name):
    norm = np.empty(img.shape, img.dtype)
    norm[..., 0] = img[..., 0] / 360 * 255  # 0, 360
    norm[..., 1] = img[..., 1] * 255  # 0, 1
    norm[..., 2] = img[..., 2] * 255  # 0, 1

    h = np.zeros(img.shape, img.dtype)
    s = np.zeros(img.shape, img.dtype)
    v = np.zeros(img.shape, img.dtype)

    h[..., 0] = norm[..., 0]
    s[..., 1] = norm[..., 1]
    v[..., 2] = norm[..., 2]

    tmp = np.concatenate((h, s, v, norm), axis=1)
    tmp2 = np.concatenate((h[..., 0], s[..., 1], v[..., 2]), axis=1)
    cv2.imwrite(file_name + '_hsv.jpg', tmp)
    cv2.imwrite(file_name + '_hsv_.jpg', tmp2)


def write_hsi(img, file_name):
    norm = np.empty(img.shape, img.dtype)
    norm[..., 0] = img[..., 0] / 360 * 255  # 0, 360
    norm[..., 1] = img[..., 1] * 255  # 0, 1
    norm[..., 2] = img[..., 2] * 255  # 0, 1

    h = np.zeros(img.shape, img.dtype)
    s = np.zeros(img.shape, img.dtype)
    i = np.zeros(img.shape, img.dtype)

    h[..., 0] = norm[..., 0]
    s[..., 1] = norm[..., 1]
    i[..., 2] = norm[..., 2]

    tmp = np.concatenate((h, s, i, norm), axis=1)
    tmp2 = np.concatenate((h[..., 0], s[..., 1], i[..., 2]), axis=1)
    cv2.imwrite(file_name + '_hsi.jpg', tmp)
    cv2.imwrite(file_name + '_hsi_.jpg', tmp2)


def write_yuv(img, file_name):
    norm = np.empty(img.shape, img.dtype)
    norm[..., 0] = img[..., 0] * 255  # 0,255
    norm[..., 1:] = (img[..., 1:] + 0.5) * 255  # -127, 128

    y = np.zeros(img.shape, img.dtype)
    u = np.zeros(img.shape, img.dtype)
    v = np.zeros(img.shape, img.dtype)

    y[..., 0] = norm[..., 0]
    u[..., 1] = norm[..., 1]
    v[..., 2] = norm[..., 2]

    tmp = np.concatenate((y, u, v, norm), axis=1)
    cv2.imwrite(file_name + '_yuv.jpg', tmp)


def write_yiq(img, file_name):
    norm = np.empty(img.shape, img.dtype)
    norm[..., 0] = img[..., 0] * 255  # 0,1
    norm[..., 1] = (img[..., 1] + 0.523) / (0.523 * 2) * 255  # -0.523, 0.523
    norm[..., 2] = (img[..., 2] + 0.592) / (0.596 * 2) * 255  # -0.596, 0.596

    y = np.zeros(img.shape, img.dtype)
    i = np.zeros(img.shape, img.dtype)
    q = np.zeros(img.shape, img.dtype)

    y[..., 0] = norm[..., 0]
    i[..., 1] = norm[..., 1]
    q[..., 2] = norm[..., 2]

    tmp = np.concatenate((y, i, q, norm), axis=1)
    cv2.imwrite(file_name + '_yiq.jpg', tmp)


def write_ycbcr(img, file_name):
    y = np.zeros(img.shape, img.dtype)
    cb = np.zeros(img.shape, img.dtype)
    cr = np.zeros(img.shape, img.dtype)

    y[..., 0] = img[..., 0]  # 0, 255
    cb[..., 1] = img[..., 1]  # 0, 255
    cr[..., 2] = img[..., 2]  # 0, 255

    tmp = np.concatenate((y, cb, cr, img), axis=1)
    cv2.imwrite(file_name + '_ycbcr.jpg', tmp)


if __name__ == "__main__":
    img_name = '../imgs/2/lenaS.jpg'
    if len(sys.argv) > 1:
        img_name = sys.argv[1]
    img = cv2.imread(img_name)

    if img is None:
        print('Can not read `{}`'.format(img_name))
        exit(-1)

    os.makedirs('res', exist_ok=True)

    out_file_name = os.path.join('res', os.path.basename(img_name))
    write_rgb(img, out_file_name)
    write_cmy(lab_4.rgb_to_cmy(img), out_file_name)
    write_hsv(lab_4.rgb_to_hsv(img[:, :, ::-1]), out_file_name)
    write_hsi(lab_4.rgb_to_hsi(img[..., ::-1]), out_file_name)
    write_yuv(lab_4.rgb_to_yuv(img[..., ::-1]), out_file_name)
    write_yiq(lab_4.rgb_to_yiq(img[..., ::-1]), out_file_name)
    write_ycbcr(lab_4.rgb_to_ycbcr(img[..., ::-1]), out_file_name)
