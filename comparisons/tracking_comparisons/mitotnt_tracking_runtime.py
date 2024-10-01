import os.path
import tifffile
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import pandas as pd
import ome_types
from mitotnt import generate_tracking_inputs, network_tracking
import time
#  Need mitotnt to be installed in the environment

# Much of this code is adapted from:
#  https://github.com/pylattice/MitoTNT/tree/f1fdc0dd465881af99205e928f4093d30c54e60c


def run_mitotnt(im_dir, start_frame, end_frame, frame_interval, tracking_interval):
    # keeping defaults for sake of automation comparison, as specified:
    # https://github.com/pylattice/MitoTNT/blob/f1fdc0dd465881af99205e928f4093d30c54e60c/mitotnt_tracking_pipeline.ipynb

    input_dir = os.path.join(im_dir, 'tracking_inputs') + '/'
    if not os.path.isdir(input_dir):
        os.mkdir(input_dir)

    generate_tracking_inputs.generate(im_dir + '/', input_dir,
                                      start_frame, end_frame,
                                      node_gap_size=0)

    # specify additional directories
    output_dir = os.path.join(im_dir, 'tracking_outputs') + '/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    network_tracking.frametoframe_tracking(input_dir, output_dir, start_frame, end_frame, frame_interval,
                                           tracking_interval,
                                           cutoff_num_neighbor=8, cutoff_speed=None,
                                           graph_matching_depth=2, dist_exponent=1, top_exponent=1)
    # network_tracking.gap_closing(input_dir, output_dir, start_frame, end_frame, tracking_interval,
    #                              min_track_size=4, max_gap_size=3, block_size_factor=1)


if __name__ == "__main__":
    visualize = False
    # top_dirs = ["/Users/austin/GitHub/nellie-simulations/motion/for_vis"]
    top_dir = '/Users/austin/test_files/time_stuff/Mitograph'
    subdirs = os.listdir(top_dir)
    subdirs = [os.path.join(top_dir, subdir) for subdir in subdirs if os.path.isdir(os.path.join(top_dir, subdir))]
    subdirs.sort()
    mitotnt_dump_dir = '/Users/austin/test_files/time_stuff/mitotnt_dump'
    viewer = None

    # get metadata
    num_frames = 2
    lateral_px_size = 0.0655
    axial_px_size = 0.25
    frame_interval = 4.536
    tracking_interval = 1
    start_frame = 0
    end_frame = num_frames - 1

    for subdir in subdirs:

        # try:
        start_time = time.time()
        run_mitotnt(subdir, start_frame, end_frame, frame_interval, tracking_interval)
        print(f"Processing {subdir} took \n\t\t{time.time() - start_time}")
        # except:
        #     print(f"Failed on {subdir}")
        #     continue

