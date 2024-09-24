import os
import pandas as pd
from scipy import stats
import datetime
import numpy as np

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

file_dir = '/Users/austin/test_files/er_stuff'
file_prefix_1 = 'hFB12_ER_3_MMStack_Default.ome-TCYX-T1p0_Y0p0655_X0p0655-ch1-t0_to_5'
file_prefix_2 = 'u2os-not_RPE1-ER_3_MMStack_Default.ome-TCYX-T1p0_Y0p0655_X0p0655-ch1-t0_to_5'

def get_branch_features(file_prefix, t_analysis=[1, 2, 3]):
    branch_features = os.path.join(file_dir, f'{file_prefix}-features_branches.csv')
    branch_df_og = pd.read_csv(branch_features)

    # could look at 3 datapoints from image averages as well

    df_to_use = branch_df_og.copy()

    ignore_cols = ['t', 'label', 'reassigned_label_raw', 'x_raw', 'y_raw', 'z_raw']
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

t_dfs_1 = get_branch_features(file_prefix_1)
t_dfs_2 = get_branch_features(file_prefix_2)

# combine all the dfs into 1 big df
combined_dfs = [t_dfs_1, t_dfs_2]
combined_dfs = [pd.concat(t_dfs, axis=0) for t_dfs in combined_dfs]
combined_dfs = pd.concat(combined_dfs, axis=0)

t_dfs_1_single = t_dfs_1[0]
t_dfs_2_single = t_dfs_2[0]

# train a random forest classifier to predict which dataset the data came from
def make_rf_model(dataset_0, dataset_1=None, save_name='', visualize=False):
    if dataset_1 is None:
        # make dataset_1 the same as dataset_0 but frames shifted by 1
        dataset_1 = dataset_0[1:] + [dataset_0[0]]

    fprs = []
    tprs = []
    aucs = []
    for frame in range(len(dataset_0)):
        dataset_0_frame = dataset_0[frame]
        dataset_1_frame = dataset_1[frame]
        # add a column to each df that is the dataset
        dataset_0_frame['dataset'] = 0
        dataset_1_frame['dataset'] = 1
        combined_dfs = pd.concat([dataset_0_frame, dataset_1_frame], axis=0)

        X = combined_dfs.drop(columns='dataset')
        y = combined_dfs['dataset']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print(f'Accuracy: {accuracy}')

        # best features
        feature_importances = clf.feature_importances_
        feature_importances = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        print(feature_importances)

        # plot roc
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)

        if visualize:
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc='lower right')
            plt.show()
    # save fpr, tpr in one csv
    fpr_tpr_df = pd.DataFrame()
    for i, (fpr, tpr) in enumerate(zip(fprs[0], tprs[0])):
        fpr_tpr_df.loc[i, 'fpr'] = fpr
        fpr_tpr_df.loc[i, 'tpr'] = tpr

    fpr_tpr_df.to_csv(f'{dt}-{save_name}_fpr_tpr.csv')

    # save aucs
    aucs_df = pd.DataFrame(aucs, columns=['auc'])
    aucs_df.to_csv(os.path.join(file_dir, f'{dt}-{save_name}_aucs.csv'))

make_rf_model(t_dfs_1, t_dfs_2, save_name='inter')
make_rf_model(t_dfs_1, save_name='intra1')
make_rf_model(t_dfs_2, save_name='intra2')

# pca of features
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X = combined_dfs.copy()
# X = X.drop(columns='dataset')
X_normalized = (X - X.mean()) / X.std()
pca.fit(X_normalized)
datasets = [t_dfs_1[0], t_dfs_1[1], t_dfs_1[2], t_dfs_2[0], t_dfs_2[1], t_dfs_2[2]]
labels = ['a_t1', 'a_t2', 'a_t3', 'b_t1', 'b_t2', 'b_t3']

results = []
for dataset in datasets:
    dataset_new = dataset.copy()
    if 'dataset' in dataset_new.columns:
        dataset_new = dataset_new.drop(columns='dataset')
    # drop nans
    dataset_new = dataset_new.dropna()

    # # check if normally distributed
    # for col in dataset_new.columns:
    #     test_data = dataset_new[col]
    #     # shift everything to be positive
    #     if test_data.min() < 0:
    #         log_test_data = test_data - test_data.min()
    #     else:
    #         log_test_data = test_data
    #     log_test_data = np.log(log_test_data)
    #     w, _ = stats.shapiro(test_data)
    #     w_log, _ = stats.shapiro(log_test_data)
    #
    #     if w_log > w:
    #         dataset_new[col] = log_test_data

    dataset_normalized = (dataset_new - dataset_new.mean()) / dataset_new.std()
    pca_results = pca.transform(dataset_normalized)
    results.append(pca_results)

# get average of each dataset
results_mean = [np.mean(result, axis=0) for result in results]
results_median = [np.median(result, axis=0) for result in results]

# plot the means with their labels
plt.figure()
for i, result in enumerate(results_median):
    plt.scatter(result[0], result[1], label=labels[i])
plt.legend()
plt.show()

# plot all the points
plt.figure()
for i, result in enumerate(results):
    plt.scatter(result[:, 0], result[:, 1], label=labels[i], alpha=0.01, s=1)
plt.legend()
plt.show()
