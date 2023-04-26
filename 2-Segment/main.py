import os
import shutil

# 317
test_labels = []
PATCH_PATH = '...'
mag = 20
multiple=40//10

def tumor(ptid, slide, multiple, mag):
    print(ptid, slide)
    tumor_path = f'{PATCH_PATH}/{ptid}/{slide}/{mag}_tumor'
    if os.path.exists(tumor_path):
        shutil.rmtree(tumor_path)
    os.makedirs(tumor_path)

    for i in os.listdir(f'{PATCH_PATH}/{ptid}/{slide}/{mag}'):
        x, y = list(map(int, i.split('.')[0].split('_')))
        c = 0
        for j in range(x*multiple, x*multiple+multiple):
            for k in range(y*multiple, y*multiple+multiple):
                 c += os.path.exists(f'{PATCH_PATH}/{ptid}/{slide}/40_tumor/{j}_{k}.jpeg')
        if c >= multiple*multiple*0.5:
            os.symlink(
                os.path.join(f'../{mag}', i),
                os.path.join(tumor_path, i)
            )
import json
with open('../metadata.json') as f:
    metadata = json.load(f)

ptids = sorted(metadata.keys())
for ptid in test_labels:
    if ptid not in metadata:
        continue
    for s in metadata[ptid]['slides'].keys():
        if metadata[ptid]['slides'][s]['is_cancer']:
            try:
                tumor(ptid, s, multiple, mag)
            except Exception as e :
                print(ptid)
