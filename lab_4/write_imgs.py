import sys
import os
import cv2
import lab_4
import numpy as np

def write_rgb(img, file_name):
    r = np.zeros(img.shape, img.dtype)
    b = np.zeros(img.shape, img.dtype)
    g = np.zeros(img.shape, img.dtype)
    r[:,:,0] = img[:, :, 0]  # 0, 255
    g[:,:,1] = img[:, :, 1]  # 0, 255
    b[:,:,2] = img[:, :, 2]  # 0, 255

    tmp = np.append(r,g, axis=1)
    tmp = np.append(tmp,b, axis=1)
    cv2.imwrite(file_name + '_rgb.jpg', np.append(tmp, img, axis=1))
    cv2.imwrite(file_name + '_Rgb.jpg', r)
    cv2.imwrite(file_name + '_rGb.jpg', g)
    cv2.imwrite(file_name + '_rgB.jpg', b)

def write_cmy(img, file_name):
    c = np.zeros(img.shape, img.dtype)
    m = np.zeros(img.shape, img.dtype)
    y = np.zeros(img.shape, img.dtype)

    c[:, :, 0] = img[:, :, 0]  # 0, 255
    m[:, :, 1] = img[:, :, 1]  # 0, 255
    y[:, :, 2] = img[:, :, 2]  # 0, 255

    tmp = np.append(c,m, axis=1)
    tmp = np.append(tmp,y, axis=1)
    cv2.imwrite(file_name + '_cmy.jpg', np.append(tmp, img, axis=1) )
    cv2.imwrite(file_name + '_Cmy.jpg', c)
    cv2.imwrite(file_name + '_cMy.jpg', m)
    cv2.imwrite(file_name + '_cmY.jpg', y)


def write_hsv(img, file_name):
    h = np.zeros(img.shape, img.dtype)
    s = np.zeros(img.shape, img.dtype)
    v = np.zeros(img.shape, img.dtype)


    h[:,:,0] = img[:, :, 0] /360 * 255  # 0, 360
    s[:,:,1] = img[:, :, 1] * 255  # 0, 1
    v[:,:,2] = img[:, :, 2] *255 # 0, 1

    tmp = np.append(h,s, axis=1)
    tmp = np.append(tmp,v, axis=1)
    cv2.imwrite(file_name + '_hsv.jpg', np.append(tmp, img, axis= 1))
    cv2.imwrite(file_name + '_Hsv.jpg', h)
    cv2.imwrite(file_name + '_hSv.jpg', s )
    cv2.imwrite(file_name + '_hsV.jpg', v )

def write_hsi(img, file_name):
    h = np.zeros(img.shape, img.dtype)
    s = np.zeros(img.shape, img.dtype)
    v = np.zeros(img.shape, img.dtype)


    h[:,:,0] = img[:, :, 0] / 360 * 255  # 0, 360
    s[:,:,1] = img[:, :, 1] * 255  # 0, 1
    v[:,:,2] = img[:, :, 2] * 255 # 0, 1

    tmp = np.append(h,s, axis=1)
    tmp = np.append(tmp,v, axis=1)
    cv2.imwrite(file_name + '_hsi.jpg', np.append(tmp, img, axis= 1))
    cv2.imwrite(file_name + '_Hsi.jpg', h )
    cv2.imwrite(file_name + '_hSi.jpg', s )
    cv2.imwrite(file_name + '_hsI.jpg', i )

def write_yuv(img, file_name):
    y[:,:,0] = img[:, :, 0]  # 0, 255
    u[:,:,1] = img[:, :, 1]  # -128, 127
    v[:,:,2] = img[:, :, 2]  # -128, 127

    tmp = np.append(y,u, axis=1)
    tmp = np.append(tmp,v, axis=1)
    cv2.imwrite(file_name + '_yuv.jpg', np.appen(tmp, img, axis=1))
    cv2.imwrite(file_name + '_Yuv.jpg', y * 255)
    cv2.imwrite(file_name + '_yUv.jpg', (u + 0.5) * 255)
    cv2.imwrite(file_name + '_yuV.jpg', (v + 0.5) * 255)


def write_yiq(img, file_name):
    y[:,:,0] = img[:, :, 0]  # 0, 1
    i[:,:,1] = img[:, :, 1]  # -0.523, 0.523
    q[:,:,2] = img[:, :, 2]  # -0.596, 0.596

    tmp = np.append(y,i, axis=1)
tmp = np.append(tmp,q, axis=1)
    cv2.imwrite(file_name + '_yiq.jpg', np.append(tmp, img, axis=1))
    cv2.imwrite(file_name + '_Yiq.jpg', y * 255)
    cv2.imwrite(file_name + '_yIq.jpg', (i + 0.523) / (0.523 * 2) * 255)
    cv2.imwrite(file_name + '_yiQ.jpg', (q + 0.592) / (0.596 * 2) * 255)


def write_ycbcr(img, file_name):
    y[:,:,0] = img[:, :, 0]  # 0, 1
    cb[:,:,1] = img[:, :, 1]  # 0, 1
    cr[:,:,2] = img[:, :, 2]  # 0, 1

    tmp = np.append(y,cb, axis=1)
    tmp = np.append(tmp,cr, axis=1)
    cv2.imwrite(file_name + '_ycbcr.jpg', np.append(tmp,img, axis=1))
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
    write_rgb(img, out_file_name)
    write_cmy(lab_4.rgb_to_cmy(img), out_file_name)
    write_hsv(lab_4.rgb_to_hsv(img), out_file_name)
    write_hsi(lab_4.rgb_to_hsi(img), out_file_name)
    write_yuv(lab_4.rgb_to_yuv(img), out_file_name)
    write_yiq(lab_4.rgb_to_yiq(img), out_file_name)
    write_ycbcr(lab_4.rgb_to_ycbcr(img), out_file_name)
