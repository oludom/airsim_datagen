

import cv2

import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import pdb


def read_pfm(file):
    """ Read a pfm file """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    header = str(bytes.decode(header, encoding='utf-8'))
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    pattern = r'^(\d+)\s(\d+)\s$'
    temp_str = str(bytes.decode(file.readline(), encoding='utf-8'))
    dim_match = re.match(pattern, temp_str)
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        temp_str += str(bytes.decode(file.readline(), encoding='utf-8'))
        dim_match = re.match(pattern, temp_str)
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header: width, height cannot be found')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    # DEY: I don't know why this was there.
    file.close()

    return data, scale


li = "/data/datasets/X4Gates_Circle_2/track2/image_left/image1101.png"
di = "/data/datasets/X4Gates_Circle_2/track2/image_depth/image1101.pfm"

depth = read_pfm(di)

depth = cv2.convertScaleAbs(depth[0], alpha=255/depth[0].max())
# depth = (255 - depth)
# cv2.imshow("depth", depth)
# cv2.waitKey(0)

cv2.imwrite("depth.png", depth)