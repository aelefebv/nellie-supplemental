import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import datetime
import networkx as nx
from scipy import ndimage as ndi

from tifffile import tifffile

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
pixel_im_prefix = '-im_pixel_class.ome.tif'
label_im_prefix = '-im_instance_label.ome.tif'
label_im_path_1 = os.path.join(file_dir, 'nellie_necessities', f'{file_prefix_1}{label_im_prefix}')
label_im_1 = tifffile.memmap(label_im_path_1, mode='r')[1:4]
pixel_im_path_1 = os.path.join(file_dir, 'nellie_necessities', f'{file_prefix_1}{pixel_im_prefix}')
pixel_im_1 = tifffile.memmap(pixel_im_path_1, mode='r')[1:4]

label_im_path_2 = os.path.join(file_dir, 'nellie_necessities', f'{file_prefix_2}{label_im_prefix}')
label_im_2 = tifffile.memmap(label_im_path_2, mode='r')[1:4]
pixel_im_path_2 = os.path.join(file_dir, 'nellie_necessities', f'{file_prefix_2}{pixel_im_prefix}')
pixel_im_2 = tifffile.memmap(pixel_im_path_2, mode='r')[1:4]


# get the largets 'organelle_area_raw' row from each t_df
largest_organelles_1 = []
largest_organelles_2 = []
for t_df in t_dfs_1_organelles:
    largest_organelles_1.append(t_df.loc[t_df['organelle_area_raw'].idxmax()])
for t_df in t_dfs_2_organelles:
    largest_organelles_2.append(t_df.loc[t_df['organelle_area_raw'].idxmax()])

largest_organelle_coords_1 = [np.argwhere(label_im_1[frame] == lo['label']) for frame, lo in enumerate(largest_organelles_1)]
largest_organelle_coords_2 = [np.argwhere(label_im_2[frame] == lo['label']) for frame, lo in enumerate(largest_organelles_2)]



def create_er_graph(branch_label, junction_label):
    G = nx.Graph()

    for j in range(1, junction_label.max() + 1):
        G.add_node(j, type='junction')

    for b in range(1, branch_label.max() + 1):
        branch_coords = np.argwhere(branch_label == b)
        connected_junctions = set()
        for coord in branch_coords:
            y, x = coord
            neighborhood = junction_label[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            connected_junctions.update(np.unique(neighborhood[neighborhood > 0]))

        if len(connected_junctions) == 2:
            j1, j2 = list(connected_junctions)
            G.add_edge(j1, j2, branch_id=b)

    return G


@dataclass
class GraphStats:
    filename: str
    frame: int
    num_edges: int
    num_nodes: int
    num_components: int
    normalized_cyclomatic_number: float
    avg_degree: float
    clustering_coeff: float
    avg_shortest_path: float
    max_betweenness_centrality: float
    max_degree_centrality: float

new_label_im_1 = np.zeros_like(label_im_1)
new_label_im_2 = np.zeros_like(label_im_2)
for frame, locs in enumerate(largest_organelle_coords_1):
    new_label_im_1[frame][tuple(locs.T)] = 1
for frame, locs in enumerate(largest_organelle_coords_2):
    new_label_im_2[frame][tuple(locs.T)] = 1

all_graph_stats = []
# for both sets of largest organelles
for new_label_im, pixel_im, file_prefix in zip([new_label_im_1, new_label_im_2], [pixel_im_1, pixel_im_2], [file_prefix_1, file_prefix_2]):
    print(f'\nAnalyzing {file_prefix}')
    # get all the branches of the largest organelle:
    branch_mask = (pixel_im == 3) & new_label_im

    branch_labels = []
    for frame in branch_mask:
        branch_label, _ = ndi.label(frame, structure=np.ones((3, 3)))
        branch_labels.append(branch_label)
    branch_labels = np.array(branch_labels)

    junction_mask = (pixel_im == 4) & new_label_im

    junction_labels = []
    for frame in junction_mask:
        junction_label, _ = ndi.label(frame, structure=np.ones((3, 3)))
        junction_labels.append(junction_label)
    junction_labels = np.array(junction_labels)

    for frame in range(len(branch_labels)):
    # for frame in range(1):
        print(f'frame {frame} of {len(branch_labels)}')
        er_graph = create_er_graph(branch_labels[frame], junction_labels[frame])
        num_edges = er_graph.number_of_edges()
        num_nodes = er_graph.number_of_nodes()
        num_components = nx.number_connected_components(er_graph)
        cyclomatic_number = num_edges - num_nodes + num_components
        max_cyclomatic = (num_nodes * (num_nodes - 1) / 2) - num_nodes + 1
        normalized_cyclomatic_number = cyclomatic_number / max_cyclomatic if max_cyclomatic > 0 else 0
        avg_degree = sum(dict(er_graph.degree()).values()) / num_nodes
        clustering_coeff = nx.average_clustering(er_graph)
        avg_shortest_path = nx.average_shortest_path_length(er_graph)
        degree_centrality = nx.degree_centrality(er_graph)
        betweenness_centrality = nx.betweenness_centrality(er_graph)
        max_betweenness_centrality = max(betweenness_centrality.values())
        max_degree_centrality = max(degree_centrality.values())

        graph_stats = GraphStats(
            filename=file_prefix, frame=frame,
            num_edges=num_edges, num_nodes=num_nodes, num_components=num_components,
            normalized_cyclomatic_number=normalized_cyclomatic_number, avg_degree=avg_degree,
            clustering_coeff=clustering_coeff, avg_shortest_path=avg_shortest_path,
            max_betweenness_centrality=max_betweenness_centrality, max_degree_centrality=max_degree_centrality
            # max_betweenness_centrality=0, max_degree_centrality=0
        )
        all_graph_stats.append(graph_stats)

output_path = os.path.join(file_dir, f'{dt}-graph_outputs.csv')
graph_stats_df = pd.DataFrame([vars(gs) for gs in all_graph_stats])
graph_stats_df.to_csv(output_path)
