import os
import pickle
import numpy as np


raw_path = r"C:\Users\austin\GitHub\nellie-simulations\px_sizes\outputs"

nellie_outputs = os.path.join(raw_path, 'nellie_output')
mitograph_outputs = os.path.join(raw_path, 'mitograph')
mitometer_outputs = os.path.join(raw_path, 'mitometer')
otsu_outputs = os.path.join(raw_path, 'otsu')
triangle_outputs = os.path.join(raw_path, 'triangle')

with open(os.path.join(nellie_outputs, 'nellie_stats.pkl'), 'rb') as f:
    nellie_stats = pickle.load(f)

with open(os.path.join(mitograph_outputs, 'mitograph_stats.pkl'), 'rb') as f:
    mitograph_stats = pickle.load(f)

with open(os.path.join(mitometer_outputs, 'mitometer_stats.pkl'), 'rb') as f:
    mitometer_stats = pickle.load(f)

with open(os.path.join(otsu_outputs, 'otsu_stats.pkl'), 'rb') as f:
    otsu_stats = pickle.load(f)

with open(os.path.join(triangle_outputs, 'triangle_stats.pkl'), 'rb') as f:
    triangle_stats = pickle.load(f)

keys = list(nellie_stats['iou'].keys())


class ValsSorted:
    def __init__(self, key):
        self.key = key
        splits = key.split('-')
        px_size = splits[1].split('_')[-1]
        length = splits[2].split('_')[-1]
        thickness = splits[3].split('_')[-1].split('.')[0]

        self.std = float(splits[0].split('_')[-1])
        self.px_size = float(px_size.replace('p', '.'))
        self.length = float(length.replace('p', '.'))
        self.thickness = float(thickness.replace('p', '.'))

        self.nellie_f1 = nellie_stats['f1'][key] if key in nellie_stats['f1'] else 0
        self.mitograph_f1 = mitograph_stats['f1'][key] if key in mitograph_stats['f1'] else 0
        self.mitometer_f1 = mitometer_stats['f1'][key] if key in mitometer_stats['f1'] else 0
        self.otsu_f1 = otsu_stats['f1'][key] if key in otsu_stats['f1'] else 0
        self.triangle_f1 = triangle_stats['f1'][key] if key in triangle_stats['f1'] else 0

        self.nellie_iou = nellie_stats['iou'][key] if key in nellie_stats['iou'] else 0
        self.mitograph_iou = mitograph_stats['iou'][key] if key in mitograph_stats['iou'] else 0
        self.mitometer_iou = mitometer_stats['iou'][key] if key in mitometer_stats['iou'] else 0
        self.otsu_iou = otsu_stats['iou'][key] if key in otsu_stats['iou'] else 0
        self.triangle_iou = triangle_stats['iou'][key] if key in triangle_stats['iou'] else 0


all_vals = []
for key in keys:
    all_vals.append(ValsSorted(key))

# save all the all_vals to a csv, where each row is a different key and each column is an attribute
import pandas as pd

df = pd.DataFrame([vars(val) for val in all_vals])
df.to_csv(os.path.join(raw_path, 'pxsizes_all_vals.csv'), index=False)
