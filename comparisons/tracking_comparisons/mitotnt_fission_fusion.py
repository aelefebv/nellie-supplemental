import os
import pandas as pd
import numpy as np

from mitotnt.detect_fusion_fission import detect
mitotnt_dir = '/Users/austin/GitHub/nellie-simulations/motion/fission_fusion/outputs/mitoTNT'
all_dirs = os.listdir(mitotnt_dir)
all_dirs = [os.path.join(mitotnt_dir, d) for d in all_dirs if os.path.isdir(os.path.join(mitotnt_dir, d))]

for all_dir in all_dirs:
    input_dir = all_dir + "/tracking_inputs/"
    output_dir = all_dir + "/tracking_outputs/"
    remodel_dir = output_dir

    track_csv = os.path.join(output_dir, 'final_node_tracks.csv')
    if not os.path.exists(track_csv):
        print(f"No tracks detected in {track_csv}")
        continue
    df = pd.read_csv(track_csv)
    # get the max num in the "frame_id" columns
    num_frames = df['frame_id'].max()
    if np.isnan(num_frames):
        print(f"No tracks detected in {track_csv}")
        continue

    half_min_window_size = 2  # The minimum we can set without running into the "frame_id" issue every time
    start_frame = half_min_window_size
    end_frame = num_frames - half_min_window_size
    nope = False
    while not nope:
        try:
            detect(input_dir, output_dir, remodel_dir, start_frame, end_frame, half_win_size=half_min_window_size)
            nope = True
        except KeyError as e:
            print(f"Failed on {track_csv}")
            print(e)
            half_min_window_size += 1
            start_frame = half_min_window_size
            end_frame = num_frames - half_min_window_size
            if start_frame >= end_frame:
                print(f"Failed on {track_csv}")
                break
            continue
