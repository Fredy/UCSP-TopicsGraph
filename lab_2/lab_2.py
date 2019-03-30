import os
import cv2
import numpy as np

def _def_operation(window, *args, **kwargs):
    return round(window.sum() / window.size)

def img_filter(src, window_size, operation=_def_operation):
    rows, cols = img.shape[:2]

    out = np.empty(src.shape, src.dtype)


    step = (window_size - 1) // 2
    for i in range(rows):
        for j in range(cols):
            # in intervals everything that is less or equals to -rows == 0
            # and everythin that is greater or equals to rows == rows
            r0 = i - step - rows
            r1 = i + step
            c0 = j - step - cols
            c1 = j + step
            window = src[r0:r1, c0:c1]
            out[i,j] = operation(window)
    
    return out



if __name__ == "__main__":
    img = cv2.imread('../imgs/woman.jpg', 0)

    os.makedirs('res', exist_ok=True)


    cv2.imwrite('res/filter_1.jpg', img_filter(img, 9))
