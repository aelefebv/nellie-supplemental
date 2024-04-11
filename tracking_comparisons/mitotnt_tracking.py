import os.path
import tifffile
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import pandas as pd
import ome_types

from mitotnt import generate_tracking_inputs, network_tracking


def convert(im_dir, im_name):
    im = tifffile.imread(os.path.join(im_dir, im_name)).astype(np.uint16)
    for i, frame in enumerate(im):
        new_dir = os.path.join(im_dir, f"frame_{i}")
        os.makedirs(new_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(new_dir, f"frame_{i}.tif"), frame)
    return im


def run_mitograph(frame_dir, lateral_pixel_size, axial_pixel_size, mitograph_dir):
    command = f"{mitograph_dir}/MitoGraph -xy {lateral_pixel_size} -z {axial_pixel_size} -path {frame_dir}/"
    print(command)
    os.system(command)


def parallel_mitograph(im_dir, lateral_pixel_size, axial_pixel_size, mitograph_dir):
    num_processes = cpu_count()

    all_frames = sorted(os.listdir(im_dir))
    all_frames = [os.path.join(im_dir, frame) for frame in all_frames if os.path.isdir(os.path.join(im_dir, frame))]
    tasks = [(frame, lateral_pixel_size, axial_pixel_size, mitograph_dir) for frame in all_frames]

    for i in range(0, len(tasks), num_processes):
        with Pool(num_processes) as p:
            p.starmap(run_mitograph, tasks[i:i + num_processes])


def run_mitotnt(im_dir, start_frame, end_frame, frame_interval, tracking_interval):
    # keeping defaults for sake of automation comparison.

    input_dir = os.path.join(im_dir, 'tracking_inputs')
    if not os.path.isdir(input_dir):
        os.mkdir(input_dir)

    generate_tracking_inputs.generate(im_dir, input_dir,
                                      start_frame, end_frame,
                                      node_gap_size=0)

    # specify additional directories
    output_dir = im_dir + 'tracking_outputs/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    network_tracking.frametoframe_tracking(input_dir, output_dir, start_frame, end_frame, frame_interval,
                                           tracking_interval,
                                           cutoff_num_neighbor=8, cutoff_speed=None,
                                           graph_matching_depth=2, dist_exponent=1, top_exponent=1)
    network_tracking.gap_closing(input_dir, output_dir, start_frame, end_frame, tracking_interval,
                                 min_track_size=4, max_gap_size=3, block_size_factor=1)


if __name__ == "__main__":
    im_dir = "/Users/austin/Downloads/sim2/"
    im_name = "multi_length-1.ome.tif"
    mitograph_dir = "/Users/austin/Desktop/MitoGraph"
    visualize = True

    im = convert(im_dir, im_name)

    # get metadata
    num_frames = len(im)
    ome_xml = tifffile.tiffcomment(os.path.join(im_dir, im_name))
    ome = ome_types.from_xml(ome_xml)
    lateral_px_size = ome.images[0].pixels.physical_size_x
    axial_px_size = ome.images[0].pixels.physical_size_z
    frame_interval = ome.images[0].pixels.time_increment
    tracking_interval = 1
    start_frame = 0
    end_frame = num_frames - 1

    parallel_mitograph(im_dir, lateral_px_size, axial_px_size, mitograph_dir)
    run_mitotnt(im_dir, start_frame, end_frame, frame_interval, tracking_interval)

    if visualize:
        import napari

        viewer = napari.Viewer()
        track_file = os.path.join(im_dir, 'tracking_outputs', 'final_node_tracks.csv')
        df = pd.read_csv(track_file)

        napari_tracks = []
        # each row is a track, go through each row and add it to the array
        for i, row in df.iterrows():
            napari_tracks.append([row['unique_node_id'], row['frame_id'], row['z'], row['y'], row['x']])

        viewer = napari.Viewer()
        viewer.add_tracks(napari_tracks)
        # for some reason, y is flipped in the mitotnt outputs... could be graph-like coords vs image coords
        viewer.add_image(im[:, :, ::-1, ...], name='im', scale=(axial_px_size, lateral_px_size, lateral_px_size))
