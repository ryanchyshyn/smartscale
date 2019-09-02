#!/usr/bin/python3

import time
import cv2
import numpy
import argparse
import copy
from PIL import Image
import os
import imageio

def get_perspective_matrixes(w, h):
    src_points = numpy.float32(
        [[w * 0.301, h * 0.883],
         [w * 0.329, h * 0.25],
         [w * 0.831, h * 0.364],
         [w * 0.842, h * 0.854]])
    dst_points = numpy.float32(
        [[0, h],
         [0, 0],
         [w, 0],
         [w, h]])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)

    return M, Minv


def warp_image(img):
    img_size = (img.shape[1], img.shape[0])

    M, Minv = get_perspective_matrixes(img.shape[1], img.shape[0])

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_AREA)  # keep same size as input image
    return warped

dirs = [f for f in os.listdir(".") if not os.path.isfile(f)]
for dir in dirs:
    print("Directory " + dir)
    for file in [f for f in os.listdir(dir) if not f.startswith("_")]:
        print("File " + file)
        image_file = os.path.join(dir, file)
        image = imageio.imread(image_file)
        image = warp_image(image)
        Image.fromarray(image).save(os.path.join(dir, "_" + file))
