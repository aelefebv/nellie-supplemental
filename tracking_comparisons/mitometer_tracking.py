from scipy.optimize import linear_sum_assignment
from tifffile import tifffile
import ome_types
import numpy as np
from skimage import measure
# try:
#     import cupy as xp
#     import cupyx.scipy.ndimage as ndi
#     device_type = 'cuda'
# except ImportError:
xp = np
import scipy.ndimage as ndi
device_type = 'cpu'
import os


def get_delta_matrix(tracks, mitos, attribute):
    num_mito = len(mitos)
    num_tracks = len(tracks)

    delta_matrix = np.zeros((num_mito, num_tracks))
    mito_attribute_vector = np.array([getattr(mitos[i], attribute) for i in range(num_mito)])
    track_attribute_vector = np.array([getattr(tracks[i]['mitos'][-1], attribute) for i in range(num_tracks)])

    for i in range(num_tracks):
        # difference between the mitochondria and the tracks divided by the track attribute
        delta_matrix[:, i] = np.abs(mito_attribute_vector - track_attribute_vector[i]) / track_attribute_vector[i]

    return delta_matrix


def get_frame_matrix(tracks, mitos):
    num_mito = len(mitos)
    num_tracks = len(tracks)

    frame_matrix = np.zeros((num_mito, num_tracks))
    mito_frame_vector = np.array([mitos[i].frame for i in range(num_mito)])
    track_frame_vector = np.array([tracks[i]['frames'][-1] for i in range(num_tracks)])

    for i in range(num_tracks):
        frame_matrix[:, i] = mito_frame_vector - track_frame_vector[i]

    return frame_matrix

distance_thresh_um = 1
frame_thresh = 3

top_dir = r"C:\Users\austin\GitHub\nellie-simulations\motion\linear"
all_files = os.listdir(top_dir)
tif_files = [file for file in all_files if file.endswith('.tif')]
for file in tif_files[:1]:
    # basename = os.path.basename(file)
    noiseless_path = os.path.join(top_dir, 'noiseless', file)
    mask_im = tifffile.imread(noiseless_path) > 0

    im_path = os.path.join(top_dir, file)
    im = tifffile.imread(os.path.join(top_dir, file))
    ome_xml = tifffile.tiffcomment(im_path)
    ome = ome_types.from_xml(ome_xml)

    dim_sizes = {'X': ome.images[0].pixels.physical_size_x, 'Y': ome.images[0].pixels.physical_size_y,
                 'Z': ome.images[0].pixels.physical_size_z, 'T': ome.images[0].pixels.time_increment}

    vel_thresh_um = distance_thresh_um * dim_sizes['T']

    weights = {'vol': 1, 'majax': 1, 'minax': 1, 'z_axis': 1, 'solidity': 1, 'surface_area': 1, 'intensity': 1}

    num_frames = im.shape[0]
    # num_frames = 2

    frame_mito = {}
    tracks = []
    for frame in range(num_frames):
        label_im, num_labels = ndi.label(mask_im[frame])
        frame_mito[frame] = measure.regionprops(label_im, intensity_image=im[frame], spacing=(dim_sizes['Z'], dim_sizes['Y'], dim_sizes['X']))
        for mito in frame_mito[frame]:
            # get surface area
            v, f, _, _ = measure.marching_cubes(mito.intensity_image > 0, spacing=(dim_sizes['Z'], dim_sizes['Y'], dim_sizes['X']))
            mito.surface_area = measure.mesh_surface_area(v, f)
            mito.frame = frame
            if frame == 0:
                tracks.append({'mitos': [mito], 'frames': [frame]})

    running_confidence_costs = []
    for frame in range(num_frames):
        track_centroids = np.array([track['mitos'][-1].centroid for track in tracks])
        frame_centroids = np.array([mito.centroid for mito in frame_mito[frame]])
        distance_matrix = np.linalg.norm(track_centroids[:, None] - frame_centroids, axis=-1).T
        volume_matrix = get_delta_matrix(tracks, frame_mito[frame], 'area') ** 2
        majax_matrix = get_delta_matrix(tracks, frame_mito[frame], 'major_axis_length') ** 2
        minax_matrix = get_delta_matrix(tracks, frame_mito[frame], 'minor_axis_length') ** 2
        z_axis_matrix = get_delta_matrix(tracks, frame_mito[frame], 'equivalent_diameter') ** 2
        solidity_matrix = get_delta_matrix(tracks, frame_mito[frame], 'solidity') ** 2
        surface_area_matrix = get_delta_matrix(tracks, frame_mito[frame], 'surface_area') ** 2
        intensity_matrix = get_delta_matrix(tracks, frame_mito[frame], 'mean_intensity') ** 2
        frame_matrix = get_frame_matrix(tracks, frame_mito[frame])

        distance_matrix = np.where(distance_matrix > vel_thresh_um, np.nan, distance_matrix)
        distance_matrix = np.where(np.abs(frame_matrix) > frame_thresh, np.nan, distance_matrix) ** 2

        # z score normalize
        distance_matrix_z = (distance_matrix - np.nanmean(distance_matrix)) / np.nanstd(distance_matrix) if np.nanstd(distance_matrix) != 0 else distance_matrix * 0
        volume_matrix_z = (volume_matrix - np.nanmean(volume_matrix)) / np.nanstd(volume_matrix) if np.nanstd(volume_matrix) != 0 else volume_matrix * 0
        majax_matrix_z = (majax_matrix - np.nanmean(majax_matrix)) / np.nanstd(majax_matrix) if np.nanstd(majax_matrix) != 0 else majax_matrix * 0
        minax_matrix_z = (minax_matrix - np.nanmean(minax_matrix)) / np.nanstd(minax_matrix) if np.nanstd(minax_matrix) != 0 else minax_matrix * 0
        z_axis_matrix_z = (z_axis_matrix - np.nanmean(z_axis_matrix)) / np.nanstd(z_axis_matrix) if np.nanstd(z_axis_matrix) != 0 else z_axis_matrix * 0
        solidity_matrix_z = (solidity_matrix - np.nanmean(solidity_matrix)) / np.nanstd(solidity_matrix) if np.nanstd(solidity_matrix) != 0 else solidity_matrix * 0
        surface_area_matrix_z = (surface_area_matrix - np.nanmean(surface_area_matrix)) / np.nanstd(surface_area_matrix) if np.nanstd(surface_area_matrix) != 0 else surface_area_matrix * 0
        intensity_matrix_z = (intensity_matrix - np.nanmean(intensity_matrix)) / np.nanstd(intensity_matrix) if np.nanstd(intensity_matrix) != 0 else intensity_matrix * 0

        cost_matrix = (weights['vol'] * volume_matrix_z + weights['majax'] * majax_matrix_z +
                       weights['minax'] * minax_matrix_z + weights['z_axis'] * z_axis_matrix_z +
                       weights['solidity'] * solidity_matrix_z + weights['surface_area'] * surface_area_matrix_z +
                       weights['intensity'] * intensity_matrix_z) + distance_matrix_z

        if len(running_confidence_costs) == 0:
            start_cost = 1
        else:
            start_cost = xp.nanquantile(running_confidence_costs, 0.98)

        # diagonal of start costs of size of the cost matrix
        new_track_matrix = np.zeros((len(frame_mito[frame]), len(frame_mito[frame]))) + np.nan
        new_track_matrix[np.diag_indices_from(new_track_matrix)] = start_cost

        assign_matrix = np.hstack((cost_matrix, new_track_matrix))
        assign_matrix[xp.isnan(assign_matrix)] = 100
        # get min local cost assignments
        min_local = xp.argmin(assign_matrix, axis=1)

        # solve LAP (get min global cost assignments)
        row_ind, col_ind = linear_sum_assignment(assign_matrix)

        # add matched mito from LAP to track
        for i, j in zip(row_ind, col_ind):
            if j < len(tracks):
                tracks[j]['mitos'].append(frame_mito[frame][i])
                tracks[j]['frames'].append(frame)
            else:
                tracks.append({'mitos': [frame_mito[frame][i]], 'frames': [frame]})

        # check where local and global assignments match
        confident_assignments = xp.where(min_local[row_ind] == col_ind)[0]

        confident_costs = assign_matrix[row_ind[confident_assignments], col_ind[confident_assignments]]
        running_confidence_costs.extend(confident_costs)



