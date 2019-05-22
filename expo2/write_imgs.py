import cv2


def write_hsv(img, file_name):
    h = img[:, :, 0]  # 0, 360
    s = img[:, :, 1]  # 0, 1
    v = img[:, :, 2]  # 0, 1

    cv2.imwrite(file_name + '_H.jpg', h / 360 * 255)
    cv2.imwrite(file_name + '_S.jpg', s * 255)
    cv2.imwrite(file_name + '_V.jpg', v * 255)

def write_yuv(img, file_name):
    y = img[:, :, 0]  # 0, 255
    u = img[:, :, 1]  # -128, 127
    v = img[:, :, 2]  # -128, 127

    cv2.imwrite(file_name + '_Y.jpg', y * 255)
    cv2.imwrite(file_name + '_U.jpg', (u + 0.5) * 255)
    cv2.imwrite(file_name + '_V.jpg', (v + 0.5) * 255)