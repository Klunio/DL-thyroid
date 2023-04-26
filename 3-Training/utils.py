# _*_ coding: utf-8 _*_
"""
Time:     2023/04/25
Author:   Yunquan Gu(Clooney)
Version:  V 0.1
File:     utils.py.py
Describe: Write during the internship at Hikvison, Github link: https://github.com/Deeachain/GraphEmbeddings
"""
import os

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ColorList = {
    'black': [np.array([0, 0, 0]), np.array([180, 255, 46])],
    'white': [np.array([0, 0, 221]), np.array([180, 30, 255])],
    'red': [np.array([156, 43, 46]), np.array([180, 255, 255])],
    'red2': [np.array([0, 43, 46]), np.array([10, 255, 255])],
    'orange': [np.array([11, 43, 46]), np.array([25, 255, 255])],
    'yellow': [np.array([26, 43, 46]), np.array([34, 255, 255])],
    'green': [np.array([35, 43, 46]), np.array([77, 255, 255])],
    'cyan': [np.array([78, 43, 46]), np.array([99, 255, 255])],
    'blue': [np.array([100, 43, 46]), np.array([124, 255, 255])],
    'purple': [np.array([125, 43, 46]), np.array([155, 255, 255])]
}

"""
Find the main color of an image, refer: https://blog.csdn.net/wangdongwei0/article/details/89637748
"""


def get_color(frame: PIL.Image):
    hsv = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2HSV)
    maximum = -100
    color = None

    for d in ColorList:
        mask = cv2.inRange(hsv, ColorList[d][0], ColorList[d][1])
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        cnts, hiera = cv2.findContours(
            binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        if sum > maximum:
            maximum = sum
            color = d
    return color


"""
Displace tiled result of one slide
"""


def joint_patch(path, patch_size=32):
    patches = os.listdir(path)
    width = height = -1
    for p in patches:
        y, x = list(map(int, p.split('.')[0].split('_')))
        width, height = max(width, x), max(height, y)
    print('width', width, 'height', height)

    arr = np.ones([(width + 1) * patch_size, (height + 1) * patch_size, 3])
    arr.fill(255)
    for p in patches:
        y, x = list(map(int, p.split('.')[0].split('_')))
        arr[x * patch_size:x * patch_size + patch_size, y * patch_size:y * patch_size + patch_size] = \
            np.array(Image.open(os.path.join(path, p)).resize(
                (patch_size, patch_size), Image.ANTIALIAS))
    return width, height, arr


def show_joint_image(ptid, slide_name, mag='5'):
    path = f'.../{ptid}/{slide_name}/{mag}'
    width, height, im = joint_patch(path)
    plt.figure(figsize=(8, 8))
    plt.imshow(im.astype(np.uint8))
    plt.xticks(range(0, height * 32 + 32, 32))
    plt.yticks(range(0, width * 32 + 32, 32))
    plt.grid()
    plt.show()
    plt.close()
    return im
