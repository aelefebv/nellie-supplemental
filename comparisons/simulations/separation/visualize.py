import os
import pickle
import numpy as np


raw_path = r"C:\Users\austin\GitHub\nellie-simulations\separation\separation"

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

keys = list(mitograph_stats['f1'].keys())


class ValsSorted:
    def __init__(self, key):
        self.key = key
        splits = key.split('-')
        px_size = splits[1].split('_')[-1]
        length = splits[2].split('_')[-1]
        thickness = splits[3].split('_')[-1]
        separation = splits[4].split('_')[-1]

        self.std = float(splits[0].split('_')[-1])
        self.px_size = float(px_size.replace('p', '.'))
        self.length = float(length.replace('p', '.'))
        self.thickness = float(thickness.replace('p', '.'))
        self.separation = float(separation.replace('p', '.'))

        self.nellie_f1 = nellie_stats['f1'][key]
        self.mitograph_f1 = mitograph_stats['f1'][key]
        self.mitometer_f1 = mitometer_stats['f1'][key]
        self.otsu_f1 = otsu_stats['f1'][key]
        self.triangle_f1 = triangle_stats['f1'][key]


all_vals = []
for key in keys:
    all_vals.append(ValsSorted(key))

# save all the all_vals to a csv, where each row is a different key and each column is an attribute
import pandas as pd

df = pd.DataFrame([vars(val) for val in all_vals])
df.to_csv(os.path.join(raw_path, 'separation_all_vals.csv'), index=False)
