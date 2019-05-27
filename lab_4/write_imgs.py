import sys
import os
import cv2
import lab_4


def write_cmy(img, file_name):
    c = img[:, :, 0]  # 0, 255
    m = img[:, :, 1]  # 0, 255
    y = img[:, :, 2]  # 0, 255

    cv2.imwrite(file_name + '_Cmy.jpg', c)
    cv2.imwrite(file_name + '_cMy.jpg', m)
    cv2.imwrite(file_name + '_cmY.jpg', y)


def write_hsv(img, file_name):
    # Also works for HSI
    h = img[:, :, 0]  # 0, 360
    s = img[:, :, 1]  # 0, 1
    v = img[:, :, 2]  # 0, 1

    cv2.imwrite(file_name + '_Hsv.jpg', h / 360 * 255)
    cv2.imwrite(file_name + '_hSv.jpg', s * 255)
    cv2.imwrite(file_name + '_hsV.jpg', v * 255)


def write_yuv(img, file_name):
    y = img[:, :, 0]  # 0, 255
    u = img[:, :, 1]  # -128, 127
    v = img[:, :, 2]  # -128, 127

    cv2.imwrite(file_name + '_Yuv.jpg', y * 255)
    cv2.imwrite(file_name + '_yUv.jpg', (u + 0.5) * 255)
    cv2.imwrite(file_name + '_yuV.jpg', (v + 0.5) * 255)


def write_yiq(img, file_name):
    y = img[:, :, 0]  # 0, 1
    i = img[:, :, 1]  # -0.523, 0.523
    q = img[:, :, 2]  # -0.596, 0.596

    cv2.imwrite(file_name + '_Yiq.jpg', y * 255)
    cv2.imwrite(file_name + '_yIq.jpg', (i + 0.523) / (0.523 * 2) * 255)
    cv2.imwrite(file_name + '_yiQ.jpg', (q + 0.592) / (0.596 * 2) * 255)


def write_ycbcr(img, file_name):
    y = img[:, :, 0]  # 0, 1
    cb = img[:, :, 1]  # 0, 1
    cr = img[:, :, 2]  # 0, 1

    cv2.imwrite(file_name + '_Ycbcr.jpg', y)
    cv2.imwrite(file_name + '_yCBcr.jpg', cb)
    cv2.imwrite(file_name + '_ycbCR.jpg', cr)


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
    write_cmy(lab_4.rgb_to_cmy(img), out_file_name)
    write_hsv(lab_4.rgb_to_hsv(img), out_file_name)
    write_yuv(lab_4.rgb_to_yuv(img), out_file_name)
    write_yiq(lab_4.rgb_to_yiq(img), out_file_name)
    write_ycbcr(lab_4.rgb_to_ycbcr(img), out_file_name)
