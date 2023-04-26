# _*_ coding: utf-8 _*_
"""
Time:     2023/04/25
Author:   Yunquan Gu(Clooney)
Version:  V.0.1
File:     DataPrepare/PatchGenerator.py
Describe:
    Definition of `Patch Generator;
    slide_path: The absolute path of slide file.
    dest_path: The path which the patch saved.
    ptid: The PTID of the slide.
    tile_size: Tile size.
    mag: Magnification under which the slide is tiled.
    avgBkg: Average background value above which patch is dropped.

Usage: You must indicate the `DATA_PATH` that you save metadata(CP.npy) and then run with scipts like:
    >>> from PatchGenerator import PatchGenerator
    >>> import concurrent.futures import wait
    >>> import os
    >>> DATA_PATH = '/Data/'
    >>> RAW_DATA_PATH = f'{DATA_PATH}/raw_data'
    >>> PATCHES_PATH = f'{DATA_PATH}/patches'
    >>> ptids = os.listdir(RAW_DATA_PATH)
    >>> tile_size, mag = 512, 20
    >>> for ptid in ptids:
    >>>     print('Start tiling slides from patient ', ptid)
    >>>     for slide_file in os.listdir(f'{RAW_DATA_PATH}/{ptid}'):
    >>>         if 'svs' in slide_file or 'tif' in slide_file:
    >>>             pg = PatchGenerator(slide_path=f'{RAW_DATA_PATH}/{ptid}',
    >>>                                 ptid=ptid,
    >>>                                 dest_path=PATCHES_PATH,
    >>>                                 tile_size=tile_size,
    >>>                                 mag=mag
    >>>                                 )
    >>>             pg.start()
    >>>             wait(pg.queue)
    >>>             print('\tDome.')
"""
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import ImageEnhance
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator

from utils import get_color

DATA_PATH = '...'

t = ThreadPoolExecutor(max_workers=16, thread_name_prefix='generate')
cp = np.load(f'{DATA_PATH}/cut_point.npy', allow_pickle=True).item()

with open('../metadata.json') as f:
    metadata = json.load(f)


def tile(pg, begin, row, col, slide_name):
    _tile = pg.dz.get_tile(pg.level, (row, col))
    w, h = _tile.size
    if w < 480 or h < 480:
        return

    if get_color(_tile) not in ['red', 'purple']:
        return

    temp = ImageEnhance.Brightness(_tile).enhance(1.2)
    temp = ImageEnhance.Contrast(temp).enhance(1.2)
    temp = temp.convert('L')
    bw = temp.point(lambda x: 0 if x < 230 else 1, '1')
    if np.average(bw) >= pg.avgBkg:
        return

    tile_name = '{}_{}.jpeg'.format(row - begin, col)
    _tile.save(os.path.join(pg.dest_path, pg.ptid, slide_name,
               str(pg.mag), tile_name), quality=95)


class PatchGenerator:
    def __init__(self, slide_path, dest_path, ptid, tile_size, mag=5, avgBkg=0.75):
        self.slide_path = slide_path
        self.dest_path = dest_path
        self.ptid = ptid
        self.tile_size = tile_size
        self.mag = mag
        self.avgBkg = avgBkg

        # 2. prepare generator
        self.slide = open_slide(self.slide_path)
        self.dz = DeepZoomGenerator(
            self.slide, tile_size=self.tile_size, overlap=1.)
        self.level = self.dz.level_count - int(math.log(40 / self.mag, 2)) - 1
        self.cols, self.rows = self.dz.level_tiles[self.level]
        print('\tslide size', self.cols, self.rows)

        self.queue = []

    def generate(self, begin, end, slide_name):
        if not slide_name in metadata[self.ptid]['slides']:
            return

        patch_store_path = os.path.join(
            self.dest_path,
            self.ptid,
            slide_name,
            str(self.mag)
        )
        if os.path.exists(patch_store_path):
            print('\t Already Done!')
            return

        os.makedirs(patch_store_path)
        for row in range(begin, end):
            for col in range(self.rows):
                self.queue.append(
                    t.submit(tile, self, begin, row, col, slide_name)
                )
        print('\tDone submitting %d patches' % ((end - begin) * self.rows))

    def start(self):
        if not os.path.exists(os.path.join(self.dest_path, self.ptid)):
            os.mkdir(os.path.join(self.dest_path, self.ptid))

        basename = os.path.basename(self.slide_path).split('.')[0]
        print('\tPatient[{}] cases {}'.format(self.ptid, basename))

        if len(basename.split('+')) == 2:
            name1, name2 = basename.split('+')

            cut_point = int(cp[(self.ptid, basename)] * self.cols)
            print('cut point', cut_point)

            self.generate(0, cut_point, name1)
            self.generate(cut_point, self.cols, name2)

        else:
            self.generate(0, self.dz.level_tiles[self.level][0], basename)
