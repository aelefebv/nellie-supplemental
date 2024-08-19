import os
import pandas as pd
import datetime


dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

file_dir = '/Users/austin/test_files/er_stuff'
file_prefix_1 = 'hFB12_ER_3_MMStack_Default.ome-TCYX-T1p0_Y0p0655_X0p0655-ch1-t0_to_5'
file_prefix_2 = 'u2os-not_RPE1-ER_3_MMStack_Default.ome-TCYX-T1p0_Y0p0655_X0p0655-ch1-t0_to_5'

def get_feature_features(file_prefix, t_analysis=[1, 2, 3], feature='organelles'):
    feature_features = os.path.join(file_dir, f'{file_prefix}-features_{feature}.csv')
    feature_df_og = pd.read_csv(feature_features)

    # could look at 3 datapoints from image averages as well

    df_to_use = feature_df_og.copy()

    ignore_cols = ['t', 'reassigned_label_raw', 'x_raw', 'y_raw', 'z_raw']
    ignore_cols_with = ['_std_dev', '_min', '_max', '_sum', 'structure', 'intensity']

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

t_dfs_1_organelles = get_feature_features(file_prefix_1)
t_dfs_2_organelles = get_feature_features(file_prefix_2)

# get the pixel image
# do a network analysis (nodes, junctions, etc)

# get the largets 'organelle_area_raw' row from each t_df
largest_organelles = []
for t_dfs in [t_dfs_1_organelles, t_dfs_2_organelles]:
    for t_df in t_dfs:
        largest_organelles.append(t_df.loc[t_df['organelle_area_raw'].idxmax()])

# network analysis
