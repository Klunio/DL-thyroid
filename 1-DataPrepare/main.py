import sys
import os

from PatchGenerator import PatchGenerator, metadata
from concurrent.futures import wait
import os

if __name__ == '__main__':
    RAW_DATA_PATH = '...'
    PATCHES_PATH = '...'

    labels = os.listdir(RAW_DATA_PATH)
    labels = metadata.keys()

    tile_size = 512
    mag = 40

    import logging
    logging.basicConfig(filename='history.log', level=logging.DEBUG)
    logging.info('start')

    for i, ptid in enumerate(labels):
        print('Start tile slides from patient ', ptid)
        for cases in os.listdir(os.path.join(RAW_DATA_PATH, ptid)):
            if (ptid, cases) in [('31385', '1A+1B.svs')]:
                continue
            pg = PatchGenerator(
                slide_path=os.path.join(RAW_DATA_PATH, ptid, cases),
                ptid=ptid,
                dest_path=PATCHES_PATH,
                tile_size=tile_size,
                mag=mag)
            pg.start()
            wait(pg.queue)
            print('\tdone')
        logging.info(f'[{i}/{len(labels)}] - {(i+1)/len(labels)} - {ptid}')
