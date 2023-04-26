# _*_ coding: utf-8 _*_
"""
Time:     2023/04/25
Author:   Yunquan Gu(Clooney)
Version:  V.0.1
File:     DataPrepare/GetCutPoint.py
Describe:
    Preliminary step of patch tiling, since most HCC files has more than one slice.

Usage:
    Change the `DATA_PATH` and run the script with `python GetCutPoint.py`. The record file `CP.npy` would save at
    `DATA_PATH` you assigned. The file structure would be like: (ptid, slide_name) -> cut_point(a float between 0~1)
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

import numpy as np
from PIL import Image
from PIL import ImageEnhance
from openslide import open_slide

DATA_PATH = '/group_homes/Thyroid/home/share/gputemp/Thyroid/'
RAW_DATA_PATH = f'{DATA_PATH}/raw_data/'

t = ThreadPoolExecutor(max_workers=8, thread_name_prefix='generate')


def get_cut_point(_slide):
    """
    Find the point where we are supposed to split the Image
    """
    thumbnail = _slide.get_thumbnail((1024, 1024))
    width, height = thumbnail.width, thumbnail.height
    thumbnail = ImageEnhance.Contrast(thumbnail).enhance(2.5)

    # Compress the image into a line
    line = thumbnail.resize((width, 1), Image.ANTIALIAS).convert('L')
    line = np.array(line)

    # Search the split point, in other words, the whitest point.
    # We need to start searching in the quarter in case of some blank point ahead.
    index = width // 4
    cut_point = (np.argmax(line[0][index:width - index]) + index) / width
    return cut_point


def _fn(_ptid, _slide, _suffix):
    try:
        file = open_slide(f'{RAW_DATA_PATH}/{_ptid}/{_slide}.{_suffix}')
        _cp = get_cut_point(file)
    except Exception as e:  # Some slide files may be corrupted.
        raise e
        _cp = -1
    return _ptid, _slide, _cp


if __name__ == '__main__':
    memo = np.load(f'{DATA_PATH}/cut_point.npy', allow_pickle=True).item()

    for PTID in os.listdir(RAW_DATA_PATH):
        print(PTID)
        tasks = []
        for slide in os.listdir(f'{RAW_DATA_PATH}/{PTID}/'):
            slide_name, suffix = slide.split('.')
            if suffix not in ['tif', 'svs', 'SVS', 'TIF']:
                continue
            if '+' not in slide_name:
                continue
            # Measure the cut point of the slide that we had not recorded or has corrupted.
            if (PTID, slide_name) not in memo or memo[(PTID, slide_name)] == -1:
                tasks.append(t.submit(_fn, PTID, slide_name, suffix))

        wait(tasks)
        for future in as_completed(tasks):
            ptid, slide, cp = future.result()
            memo[(ptid, slide)] = cp
            print(ptid, slide, cp)

        # save the cut-point file as 'CP.npy'
        np.save(f'{DATA_PATH}/cut_point.npy', memo)
