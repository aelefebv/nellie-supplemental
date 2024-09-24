import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_prefix = 'hFB12_ER_3_MMStack_Default.ome-TCYX-T1p0_Y0p0655_X0p0655-ch1-t0_to_5'
file_dir = '/Users/austin/test_files/er_stuff'

# image_features = os.path.join(file_dir, f'{file_prefix}-features_image.csv')
# image_df = pd.read_csv(image_features)
# unique_t = image_df['t'].unique()
# # only keep the rows with t[1:-1] in unique_t (avoid first and last frame since they are missing some stats)
# image_df = image_df[image_df['t'].isin(unique_t[1:-1])]

# organelle_features = os.path.join(file_dir, f'{file_prefix}-features_organelles.csv')
# organelle_df_og = pd.read_csv(organelle_features)
# organelle_df = organelle_df_og[organelle_df_og['t'].isin(unique_t[1:-1])]

branch_features = os.path.join(file_dir, f'{file_prefix}-features_branches.csv')
branch_df_og = pd.read_csv(branch_features)
unique_t = branch_df_og['t'].unique()
branch_df = branch_df_og[branch_df_og['t'].isin(unique_t[1:-1])]

node_features = os.path.join(file_dir, f'{file_prefix}-features_nodes.csv')
node_df_og = pd.read_csv(node_features)
node_df = node_df_og[node_df_og['t'].isin(unique_t[1:-1])]

voxel_features = os.path.join(file_dir, f'{file_prefix}-features_voxels.csv')
voxel_df = pd.read_csv(voxel_features)
voxel_df = voxel_df[voxel_df['t'].isin(unique_t[1:-1])]

ignore_cols = ['t', 'label', 'reassigned_label_raw', 'x_raw', 'y_raw', 'z_raw']
df_to_use = node_df
df_og_to_use = node_df_og
# remove index col (unnamed)
df_to_use = df_to_use.drop(columns=df_to_use.columns[0])
for ignore_col in ignore_cols:
    if ignore_col in df_to_use.columns:
        df_to_use = df_to_use.drop(columns=ignore_col)
        # df_og_to_use = df_og_to_use.drop(columns=ignore_col)
# df_to_use = df_to_use.drop(columns=ignore_cols)
# remove any rows with nan
df_to_use = df_to_use.dropna()
# remove any columns with all zeros
df_to_use = df_to_use.loc[:, (df_to_use != 0).any(axis=0)]

# normalize cols
normalized_df = (df_to_use - df_to_use.mean()) / df_to_use.std()
# correlation analysis
corr = normalized_df.corr()
plt.matshow(corr)
plt.show()

# hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# calculate the distance matrix
distance_matrix = pdist(normalized_df.T)

# calculate the linkage matrix
linkage_matrix = linkage(distance_matrix, method='ward')

# plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=normalized_df.columns, orientation='top')
plt.xticks(rotation=90)

plt.show()

# replot the correlation matrix ordered by the clustering
order = [int(i) for i in dendrogram(linkage_matrix, no_plot=True)['ivl']]
plt.matshow(corr.iloc[order, order])
plt.show()

# # hdbscan
from sklearn.cluster import HDBSCAN

clusterer = HDBSCAN(min_cluster_size=5, min_samples=5)
clusterer.fit(normalized_df.T)

# tsne of features

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30)
tsne_results = tsne.fit_transform(normalized_df.T)

# plot
plt.figure()
# color by label
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusterer.labels_)
for i, txt in enumerate(normalized_df.columns):
    plt.annotate(txt, (tsne_results[i, 0], tsne_results[i, 1]))

plt.show()


# hdbscan of samples
clusterer_samples = HDBSCAN(min_cluster_size=5, n_jobs=-1)
normalized_df_t1 = (df_og_to_use[df_og_to_use['t'] == 1] - df_og_to_use[df_og_to_use['t'] == 1].mean()) / df_og_to_use[df_og_to_use['t'] == 1].std()
# drop index, t and ignored cols, and 0 and nans
normalized_df_t1 = normalized_df_t1.drop(columns=normalized_df_t1.columns[0])
for ignore_col in ignore_cols:
    if ignore_col in normalized_df_t1.columns:
        normalized_df_t1 = normalized_df_t1.drop(columns=ignore_col)
# drop cols if all values in col are na
normalized_df_t1 = normalized_df_t1.dropna(axis=1, how='all')
normalized_df_t1 = normalized_df_t1.dropna()
clusterer_samples.fit(normalized_df_t1)
# print(clusterer_samples.labels_)
# print(clusterer_samples.probabilities_)


# df_subsample = branch_df.sample(10000)
# normalized_df_subsample = (df_subsample - df_subsample.mean()) / df_subsample.std()
# subsample_labels = clusterer_samples.fit(normalized_df_subsample)
# df_subsample['cluster'] = subsample_labels.labels_
# tsne of samples
# tsne = TSNE(n_components=2, perplexity=30, n_jobs=-1)
# tsne_results_samples = tsne.fit_transform(normalized_df_t1)
# normalized_df_t1['cluster'] = clusterer_samples.labels_

# # plot
# plt.figure()
# plt.scatter(tsne_results_samples[:, 0], tsne_results_samples[:, 1], c=normalized_df_t1['cluster'])
# plt.show()


# pca of features
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_results = pca.fit_transform(normalized_df_t1)
print(pca.explained_variance_ratio_)

# plot
plt.figure()
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=clusterer_samples.labels_)
plt.show()


# plot
plt.figure()
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=normalized_df_t1['node_thickness_raw'], s=2, alpha=0.5)
plt.show()
# plot
plt.figure()
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=normalized_df_t1['lin_vel_mag_max'], s=2, alpha=0.5)
plt.show()
plt.figure()
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=normalized_df_t1['ang_vel_mag_rel_max'], s=2, alpha=0.5)
plt.show()
plt.figure()
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=normalized_df_t1['structure_mean'], s=2, alpha=0.5)
plt.show()

# find features that best correlate with the pcs
# get the loading scores
loading_scores = pca.components_

# get the most important feature for each component
most_important = [np.abs(loading_scores[i]).argmax() for i in range(loading_scores.shape[0])]
most_important_names = [normalized_df_t1.columns[most_important[i]] for i in range(loading_scores.shape[0])]
# print the most important feature
print(most_important_names)

# plot the pca results with the most important feature
plt.figure()
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=normalized_df_t1[most_important_names[0]], s=2, alpha=0.5)
plt.show()

plt.figure()
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=normalized_df_t1[most_important_names[1]], s=2, alpha=0.5)
plt.show()
