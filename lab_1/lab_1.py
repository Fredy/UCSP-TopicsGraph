import cv2
import numpy as np


def translation(src, x, y):
    out = np.zeros(src.shape, src.dtype)
    X, Y, _ = src.shape
    out[-X + x: X + x, -Y + y: Y + y] = src[-X - x: X - x, -Y - y: Y - y]
    return out


def resize(src, xfactor, yfactor):
    x, y, c = src.shape
    shape = (x // xfactor, y //yfactor, c)
    out = np.empty(shape, src.dtype)

    for 

def _scale_up(src, xfactor, yfactor):
    pass

def _scale_down(src, xfactor, yfactor):

# transformaciones geométricas
    # traslación
    # escala
    # rotación
# filtros : log, log inverso, exponencial

if __name__ == "__main__":
    img = cv2.imread('../imgs/cat_or.jpg')
    positive = translation(img, 50, 40)
    negative = translation(img, -50, -40)
    cv2.imwrite('positive.jpg', positive)
    cv2.imwrite('negative.jpg', negative)
