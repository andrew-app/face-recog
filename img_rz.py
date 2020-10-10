import cv2
import numpy as np


def img_resize(fname):

    img = cv2.imread(fname)


    width = int(300)

    height = int(375)

    dimension = (width,height)

    resize_img = cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)

    return resize_img
