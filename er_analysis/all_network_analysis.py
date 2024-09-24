import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import datetime
import networkx as nx
from scipy import ndimage as ndi
from skimage import measure

from tifffile import tifffile

dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

file_dir = '/Users/austin/test_files/er_stuff'
file_prefix = 'hFB12_ER_2_MMStack_Default.ome-CZYX-Z0p25_Y0p0655_X0p0655-ch1'

# get the pixel image
# do a network analysis (nodes, junctions, etc)
pixel_im_prefix = '-im_pixel_class.ome.tif'
label_im_prefix = '-im_instance_label.ome.tif'
label_im_path = os.path.join(file_dir, 'nellie_necessities', f'{file_prefix}{label_im_prefix}')
label_im = tifffile.memmap(label_im_path, mode='r')
pixel_im_path = os.path.join(file_dir, 'nellie_necessities', f'{file_prefix}{pixel_im_prefix}')
pixel_im = tifffile.memmap(pixel_im_path, mode='r')

largest_organelle_coords = np.argwhere(label_im > 0)

@dataclass
class GraphStats:
    filename: str
    frame: int
    num_edges: int
    num_nodes: int
    num_components: int
    normalized_cyclomatic_number: float
    avg_degree: float
    max_betweenness_centrality: float
    max_degree_centrality: float

# bincount labels to find biggest component
bincount = np.bincount(label_im.ravel())
largest_label = np.argmax(bincount[1:]) + 1
new_label_im = label_im == largest_label
#
import napari
viewer = napari.Viewer()
viewer.add_image(pixel_im, name='pixel_im')
viewer.add_labels(new_label_im, name='new_label_im')


all_graph_stats = []
# for both sets of largest organelles
# get all the branches of the largest organelle:
branch_mask = (pixel_im == 3) & new_label_im
branch_mask_all = (pixel_im == 3)

branch_label, _ = ndi.label(branch_mask, structure=np.ones((3, 3, 3)))

junction_mask = (pixel_im == 4) & new_label_im

junction_label, _ = ndi.label(junction_mask, structure=np.ones((3, 3, 3)))
junction_regions = measure.regionprops(junction_label)
junction_centroids = np.array([jr.centroid for jr in junction_regions])

expanded_branch_mask = ndi.binary_dilation(branch_mask, iterations=1)
expanded_branch_mask_all = ndi.binary_dilation(branch_mask_all, iterations=1)
viewer.add_image((pixel_im>0) & new_label_im, name='expanded_branch_mask')
viewer.add_points(junction_centroids, name='junction_centroids', face_color='red', size=5)
viewer.add_labels(pixel_im)
viewer.add_image(expanded_branch_mask)
viewer.add_image(expanded_branch_mask_all)
er_graph = nx.Graph()

junction_connections = {}
all_junction_coords = np.argwhere(junction_label > 0)
all_junction_coord_labels = junction_label[tuple(all_junction_coords.T)]
for j in range(1, junction_label.max() + 1):
    print(f'junction {j} of {junction_label.max()}')
    er_graph.add_node(j, type='junction')
    junction_coords = all_junction_coords[all_junction_coord_labels == j]
    connected_branches = set()
    for coord in junction_coords:
        z, y, x = coord
        neighborhood = branch_label[max(0, z - 1):z + 2, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        connected_branches.update(np.unique(neighborhood[neighborhood > 0]))

    junction_connections[j] = connected_branches

# add edges between junctions that are connected by the same branch
for j1 in junction_connections:
    print(f'junction {j1} of {junction_label.max()}')
    connected_branches_1 = junction_connections[j1]
    for j2 in junction_connections:
        if j1 == j2:
            continue
        connected_branches_2 = junction_connections[j2]
        common_branches = connected_branches_1.intersection(connected_branches_2)
        if len(common_branches) > 0:
            er_graph.add_edge(j1, j2, branch_id=common_branches)

#
# for b in range(1, branch_label.max() + 1):
#     branch_coords = np.argwhere(branch_label == b)
#     connected_junctions = set()
#     for coord in branch_coords:
#         z, y, x = coord
#         neighborhood = junction_label[max(0, z - 1):z + 2, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
#         connected_junctions.update(np.unique(neighborhood[neighborhood > 0]))
#
#     if len(connected_junctions) == 2:
#         j1, j2 = list(connected_junctions)
#         er_graph.add_edge(j1, j2, branch_id=b)

# er_graph = create_er_graph(branch_label, junction_label)
num_edges = er_graph.number_of_edges()
num_nodes = er_graph.number_of_nodes()
num_components = nx.number_connected_components(er_graph)
cyclomatic_number = num_edges - num_nodes + num_components
max_cyclomatic = (num_nodes * (num_nodes - 1) / 2) - num_nodes + 1
normalized_cyclomatic_number = cyclomatic_number / max_cyclomatic if max_cyclomatic > 0 else 0
degree = dict(er_graph.degree())
# bargraph of degree distribution:
import matplotlib.pyplot as plt
plt.hist(list(degree.values()))
plt.show()

# save csv of degree distribution
degree_df = pd.DataFrame(degree.items(), columns=['node', 'degree'])
degree_df.to_csv(os.path.join(file_dir, f'{dt}-degree_distribution.csv'))

# get percent of nodes over 3
num_nodes_over_3 = sum([1 for d in degree.values() if d > 3])
percent_nodes_over_3 = num_nodes_over_3 / num_nodes

# for each unique degree value, count the number of nodes with that degree
degree_counts = {}
for d in degree.values():
    if d not in degree_counts:
        degree_counts[d] = 0
    degree_counts[d] += 1

# save
degree_counts_df = pd.DataFrame(degree_counts.items(), columns=['degree', 'count'])
degree_counts_df.to_csv(os.path.join(file_dir, f'{dt}-degree_counts.csv'))

avg_degree = sum(degree.values()) / num_nodes
degree_centrality = nx.degree_centrality(er_graph)
betweenness_centrality = nx.betweenness_centrality(er_graph)
# color points based on degree centrality
colors = np.array([degree[j] for j in er_graph.nodes()])
# colors = np.array([betweenness_centrality[j] for j in er_graph.nodes()])
# colors = np.array([degree_centrality[j] for j in er_graph.nodes()])
# normalize between 0 and 1
colors = (colors - colors.min()) / (colors.max() - colors.min())# * 255
import matplotlib.pyplot as plt
colormap = plt.cm.turbo
scaled_junction_centroids = junction_centroids * (0.25, 0.0655, 0.0655)
viewer.add_points(scaled_junction_centroids, name='junction_centroids', face_color=colormap(colors), size=0.5)


max_betweenness_centrality = max(betweenness_centrality.values())
max_degree_centrality = max(degree_centrality.values())

graph_stats = GraphStats(
    filename=file_prefix, frame=0,
    num_edges=num_edges, num_nodes=num_nodes, num_components=num_components,
    normalized_cyclomatic_number=normalized_cyclomatic_number, avg_degree=avg_degree,
    max_betweenness_centrality=max_betweenness_centrality, max_degree_centrality=max_degree_centrality
    # max_betweenness_centrality=0, max_degree_centrality=0
)
all_graph_stats.append(graph_stats)

output_path = os.path.join(file_dir, f'{dt}-graph_outputs.csv')
graph_stats_df = pd.DataFrame([vars(gs) for gs in all_graph_stats])
graph_stats_df.to_csv(output_path)
