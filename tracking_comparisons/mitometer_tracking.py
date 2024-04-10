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

    # num_frames = im.shape[0]
    num_frames = 2

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

    for frame in range(num_frames):
        track_centroids = np.array([track['mitos'][-1].centroid for track in tracks])
        frame_centroids = np.array([mito.centroid for mito in frame_mito[frame]])
        distance_matrix = np.linalg.norm(track_centroids[:, None] - frame_centroids, axis=-1)
        volume_matrix = get_delta_matrix(tracks, frame_mito[frame], 'area')
        majax_matrix = get_delta_matrix(tracks, frame_mito[frame], 'major_axis_length')
        minax_matrix = get_delta_matrix(tracks, frame_mito[frame], 'minor_axis_length')
        z_axis_matrix = get_delta_matrix(tracks, frame_mito[frame], 'equivalent_diameter')
        solidity_matrix = get_delta_matrix(tracks, frame_mito[frame], 'solidity')
        surface_area_matrix = get_delta_matrix(tracks, frame_mito[frame], 'surface_area')
        intensity_matrix = get_delta_matrix(tracks, frame_mito[frame], 'mean_intensity')
        frame_matrix = get_frame_matrix(tracks, frame_mito[frame])

        distance_matrix = np.where(distance_matrix > vel_thresh_um, np.inf, distance_matrix)
        distance_matrix = np.where(np.abs(frame_matrix) > frame_thresh, np.inf, distance_matrix)
