# -*- coding: utf-8 -*-
# Create Time: 2023 04/25
# Author: Yunquan (Clooney) Gu
import json
with open('../metadata.json') as f:
    metadata = json.load(f)

train_labels = []
val_labels = []
test_labels = []

gxmu_labels = []

train_labels += gxmu_labels[:274]
val_labels += gxmu_labels[274:366]
test_labels += gxmu_labels[366:]

all_labels = train_labels + val_labels + test_labels

Folders = [
    [], [], [], [], []
]

SDH_SMU_labels = []
