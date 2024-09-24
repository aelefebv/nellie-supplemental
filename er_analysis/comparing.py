import os
import pandas as pd
from scipy import stats
import datetime
import numpy as np

dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

file_dir = '/Users/austin/test_files/er_stuff'
file_prefix_1 = 'hFB12_ER_3_MMStack_Default.ome-TCYX-T1p0_Y0p0655_X0p0655-ch1-t0_to_5'
file_prefix_2 = 'u2os-not_RPE1-ER_3_MMStack_Default.ome-TCYX-T1p0_Y0p0655_X0p0655-ch1-t0_to_5'

def get_branch_features(file_prefix):
    branch_features = os.path.join(file_dir, f'{file_prefix}-features_branches.csv')
    branch_df_og = pd.read_csv(branch_features)
    unique_t = branch_df_og['t'].unique()

    # could look at 3 datapoints from image averages as well

    df_to_use = branch_df_og.copy()

    t_analysis = [1, 2, 3]
    ignore_cols = ['t', 'label', 'reassigned_label_raw', 'x_raw', 'y_raw', 'z_raw']
    ignore_cols_with = ['_std_dev', '_min', '_max', '_sum']

    t_dfs = []
    for t in t_analysis:
        t_df = df_to_use[df_to_use['t'] == t]
        # drop first col
        t_df = t_df.drop(columns=t_df.columns[0])
        # drop ignore cols
        for ignore_col in ignore_cols:
            if ignore_col in t_df.columns:
                t_df = t_df.drop(columns=ignore_col)

        for ignore_col in ignore_cols_with:
            for col in t_df.columns:
                if ignore_col in col:
                    t_df = t_df.drop(columns=col)

        # remove any rows with nan
        t_df = t_df.dropna()
        # remove any columns with all zeros
        t_df = t_df.loc[:, (t_df != 0).any(axis=0)]
        t_dfs.append(t_df)

    return t_dfs

t_dfs_1 = get_branch_features(file_prefix_1)
t_dfs_2 = get_branch_features(file_prefix_2)

intra_variation = []
for t_dfs in [t_dfs_1, t_dfs_2]:
    # for each feature, calculate the kruskal-wallis test with multiple comparisons
    combos = [(i, j) for i in range(len(t_dfs)) for j in range(i, len(t_dfs)) if i != j]
    features = {}
    for feature in t_dfs[0].columns:
        p_vals = []
        median_diffs = []
        for i, j in combos:
            t1 = t_dfs[i][feature]
            t2 = t_dfs[j][feature]
            t_stat, p_val = stats.kruskal(t1, t2)
            if p_val < 0.05:
                p_val = 1
            else:
                p_val = 0
            # neg_log10_p_val = -1 * np.log10(p_val)
            p_vals.append(p_val)
            fold_change = np.abs(t1.median() / t2.median())
            if fold_change < 1:
                fold_change = 1 / fold_change
            # log_fold_change = np.abs(np.log2(fold_change))
            median_diffs.append(fold_change)
        features[feature] = (p_vals, median_diffs)
    intra_variation.append(features)

inter_variation = {}
for feature in t_dfs_1[0].columns:
    p_vals = []
    median_diffs = []
    for i in range(len(t_dfs_1)):
        t1 = t_dfs_1[i][feature]
        t2 = t_dfs_2[i][feature]
        t_stat, p_val = stats.kruskal(t1, t2)
        if p_val < 0.05:
            p_val = 1
        else:
            p_val = 0
        # neg_log10_p_val = -1 * np.log10(p_val)
        p_vals.append(p_val)
        fold_change = np.abs(t1.median() / t2.median())
        if fold_change < 1:
            fold_change = 1 / fold_change
        # log_fold_change = np.abs(np.log2(fold_change))
        median_diffs.append(fold_change)
    inter_variation[feature] = (p_vals, median_diffs)

# save a csv of p_vals (rows are t, columns are features) and another csv of median_diffs (rows are t, columns are features)
# intra_variation
intra_variation_p_val_df = pd.DataFrame()
intra_variation_median_diff_df = pd.DataFrame()
for i, intra_variation_dict in enumerate(intra_variation):
    feature_p_val_df = pd.DataFrame()
    feature_median_diff_df = pd.DataFrame()
    for feature, (p_vals, median_diffs) in intra_variation_dict.items():
        for i, (p_val, median_diff) in enumerate(zip(p_vals, median_diffs)):
            feature_p_val_df.loc[i, feature] = p_val
            feature_median_diff_df.loc[i, feature] = median_diff
    intra_variation_p_val_df = pd.concat([intra_variation_p_val_df, feature_p_val_df], axis=0)
    intra_variation_median_diff_df = pd.concat([intra_variation_median_diff_df, feature_median_diff_df], axis=0)

intra_variation_p_val_df.to_csv(os.path.join(file_dir, f'{dt}-intra_variation_p_vals.csv'))
intra_variation_median_diff_df.to_csv(os.path.join(file_dir, f'{dt}-intra_variation_median_diffs.csv'))

# inter_variation
inter_variation_p_val_df = pd.DataFrame()
inter_variation_median_diff_df = pd.DataFrame()
for feature, (p_vals, median_diffs) in inter_variation.items():
    for i, (p_val, median_diff) in enumerate(zip(p_vals, median_diffs)):
        inter_variation_p_val_df.loc[i, feature] = p_val
        inter_variation_median_diff_df.loc[i, feature] = median_diff

inter_variation_p_val_df.to_csv(os.path.join(file_dir, f'{dt}-inter_variation_p_vals.csv'))
inter_variation_median_diff_df.to_csv(os.path.join(file_dir, f'{dt}-inter_variation_median_diffs.csv'))

# volcano plots for both intra and inter, separately

