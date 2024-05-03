import os
import pickle
import numpy as np


raw_path = r"C:\Users\austin\GitHub\nellie-simulations\multi_grid\outputs"

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
    def __init__(self, key, intensity_val):
        self.key = key
        self.std = float(key.split('_')[-1])
        self.intensity_val = intensity_val

        self.nellie_mean_f1 = None
        self.nellie_std_f1 = None
        self.mitograph_mean_f1 = None
        self.mitograph_std_f1 = None
        self.mitometer_mean_f1 = None
        self.mitometer_std_f1 = None
        self.otsu_mean_f1 = None
        self.otsu_std_f1 = None
        self.triangle_mean_f1 = None
        self.triangle_std_f1 = None

        self.nellie_mean_iou = None
        self.nellie_std_iou = None
        self.mitograph_mean_iou = None
        self.mitograph_std_iou = None
        self.mitometer_mean_iou = None
        self.mitometer_std_iou = None
        self.otsu_mean_iou = None
        self.otsu_std_iou = None
        self.triangle_mean_iou = None
        self.triangle_std_iou = None


all_vals = {}
intensity_vector = np.linspace(6554, 58982, 10)
for key in keys:
    for intensity_val in intensity_vector:
        all_vals[f'{key}_{int(intensity_val)}'] = ValsSorted(key, intensity_val)


def get_ave_vals(stats, method_name, stat_type='f1'):
    ave_vals = {}
    for key in keys:
        for intensity_num, intensity_val in enumerate(intensity_vector):
            mean_attribute = f'{method_name}_mean_{stat_type}'
            std_attribute = f'{method_name}_std_{stat_type}'
            if key not in stats[stat_type]:
                mean_att = 0
                std_att = 0
            else:
                vals = stats[stat_type][key][:, intensity_num, :]
                mean_att = np.nanmean(vals)
                std_att = np.nanstd(vals)
            setattr(all_vals[f'{key}_{int(intensity_val)}'], mean_attribute, mean_att)
            setattr(all_vals[f'{key}_{int(intensity_val)}'], std_attribute, std_att)
    return ave_vals


ave_vals_nellie_f1 = get_ave_vals(nellie_stats, 'nellie')
ave_vals_mitograph_f1 = get_ave_vals(mitograph_stats, 'mitograph')
ave_vals_mitometer_f1 = get_ave_vals(mitometer_stats, 'mitometer')
ave_vals_otsu_f1 = get_ave_vals(otsu_stats, 'otsu')
ave_vals_triangle_f1 = get_ave_vals(triangle_stats, 'triangle')

ave_vals_nellie_iou = get_ave_vals(nellie_stats, 'nellie', 'iou')
ave_vals_mitograph_iou = get_ave_vals(mitograph_stats,'mitograph', 'iou')
ave_vals_mitometer_iou = get_ave_vals(mitometer_stats, 'mitometer', 'iou')
ave_vals_otsu_iou = get_ave_vals(otsu_stats, 'otsu', 'iou')
ave_vals_triangle_iou = get_ave_vals(triangle_stats, 'triangle', 'iou')

import pandas as pd

df = pd.DataFrame([vars(val) for val in all_vals.values()])
df.to_csv(os.path.join(raw_path, 'multigrid_all_vals.csv'), index=False)
