from tensorly.decomposition import parafac

import os
import pandas as pd
import datetime
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

file_dir = '/Users/austin/test_files/er_stuff'
file_prefix_1 = 'hFB12_ER_3_MMStack_Default.ome-TCYX-T1p0_Y0p0655_X0p0655-ch1-t0_to_5'
file_prefix_2 = 'u2os-not_RPE1-ER_3_MMStack_Default.ome-TCYX-T1p0_Y0p0655_X0p0655-ch1-t0_to_5'

def get_branch_features(file_prefix, t_analysis=[1, 2, 3]):
    branch_features = os.path.join(file_dir, f'{file_prefix}-features_branches.csv')
    branch_df_og = pd.read_csv(branch_features)

    # could look at 3 datapoints from image averages as well

    df_to_use = branch_df_og.copy()

    ignore_cols = ['label', 'reassigned_label_raw', 'x_raw', 'y_raw', 'z_raw']
    ignore_cols_with = ['_std_dev', '_min', '_max', '_sum', 'structure', 'intensity']
    # ignore_cols_with = ['structure', 'intensity']

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
t_dfs_1 = pd.concat(t_dfs_1, axis=0)

t_dfs_2 = get_branch_features(file_prefix_2)
t_dfs_2 = pd.concat(t_dfs_2, axis=0)

# # combine all the dfs into 1 big df
# combined_dfs = [t_dfs_1, t_dfs_2]
# combined_dfs = [pd.concat(t_dfs, axis=0) for t_dfs in combined_dfs]
# combined_dfs = pd.concat(combined_dfs, axis=0)
#
# t_dfs_1_single = t_dfs_1[0]
# t_dfs_2_single = t_dfs_2[0]

def preprocess_data(data_a, data_b):
    # Combine data from both cell lines
    combined_data = pd.concat([data_a, data_b], keys=['A', 'B'], names=['cell_line'])

    # Separate features from metadata
    features = combined_data.drop('t', axis=1)
    timepoints = combined_data['t']

    scaler = StandardScaler()
    normalized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)

    normalized_data = pd.concat([normalized_features, timepoints], axis=1)

    tensor = normalized_data.pivot_table(
        index=normalized_data.index.get_level_values('cell_line'),
        columns='t',
        values=[col for col in normalized_data.columns if col != 't']
    )

    tensor = tensor.values.reshape(2, 3, -1)  # 2 cell lines, 3 timepoints, N features
    return tensor, normalized_data.columns[normalized_data.columns != 't'].tolist()


def perform_tensor_decomposition(tensor, rank=3):
    factors = parafac(tensor, rank=rank, n_iter_max=1000, tol=1e-8)
    return factors

tensor, feature_names = preprocess_data(t_dfs_1, t_dfs_2)

factors_and_weights = perform_tensor_decomposition(tensor)
factors = factors_and_weights[1]

for component in range(factors[0].shape[1]):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    titles = ['Cell Lines', 'Timepoints', 'Features']
    for i, (factor, title) in enumerate(zip(factors, titles)):
        if i == 2:  # Feature factor
            axs[i].barh(range(len(feature_names)), factor[:, component])
            axs[i].set_yticks(range(len(feature_names)))
            axs[i].set_yticklabels(feature_names, fontsize=8)
        else:
            axs[i].bar(range(factor.shape[0]), factor[:, component])
            axs[i].set_xticks(range(factor.shape[0]))
            axs[i].set_xticklabels(['A', 'B'] if i == 0 else ['T1', 'T2', 'T3'])

        axs[i].set_title(f'{title} - Component {component+1}')

    plt.tight_layout()
    plt.show()

cell_line_factors = factors[0]
timepoint_factors = factors[1]
feature_factors = factors[2]

print("Cell Line Factors:")
print(cell_line_factors)
print("\nTimepoint Factors:")
print(timepoint_factors)
print("\nTop 5 Features for Component 1:")
top_features = sorted(zip(feature_names, feature_factors[:, 0]), key=lambda x: abs(x[1]), reverse=True)[:5]
for feature, weight in top_features:
    print(f"{feature}: {weight}")

# save feature names
feature_names_df = pd.DataFrame(feature_names)
feature_names_df.to_csv(os.path.join(file_dir, f'{dt}-feature_names.csv'), index=False)

# save component 1 feature weights
component_1_feature_weights = pd.DataFrame(feature_factors[:, 0], index=feature_names, columns=['weight'])
component_1_feature_weights.to_csv(os.path.join(file_dir, f'{dt}-component_1_feature_weights.csv'))

# save component 1 cell line weights
component_1_cell_line_weights = pd.DataFrame(cell_line_factors[:, 0], index=['hFB12', 'U2OS'], columns=['weight'])
component_1_cell_line_weights.to_csv(os.path.join(file_dir, f'{dt}-component_1_cell_line_weights.csv'))

# save component 1 timepoint weights
component_1_timepoint_weights = pd.DataFrame(timepoint_factors[:, 0], index=['1', '2', '3'], columns=['weight'])
component_1_timepoint_weights.to_csv(os.path.join(file_dir, f'{dt}-component_1_timepoint_weights.csv'))
