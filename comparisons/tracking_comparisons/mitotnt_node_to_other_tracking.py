from mitotnt.track_fragments_segments import track_from_nodes
import os
import pandas as pd

tracks_dir = '/Users/austin/GitHub/nellie-simulations/motion/angular/outputs/mitoTNT'
all_dirs = os.listdir(tracks_dir)
all_dirs = [os.path.join(tracks_dir, d) for d in all_dirs if os.path.isdir(os.path.join(tracks_dir, d))]

for all_dir in all_dirs:
    output_dir = os.path.dirname(all_dir)
    output_csv = os.path.join(all_dir, 'tracking_outputs', 'final_node_tracks.csv')
    try:
        df = pd.read_csv(output_csv)
    except:
        print(f"Failed to read {output_csv}")
        continue
    # find the max num in the 'frame_id' column
    num_frames = df['frame_id'].max()
    all_dir_basename = os.path.basename(all_dir).split('-')[-1].split('_')[-1]
    frame_step = float(all_dir_basename.replace('p', '.'))

    try:
        track_from_nodes(output_csv, all_dir,
                         num_frames=num_frames, start_frame=0, frame_step=frame_step,
                         level='fragment', voxel_size=0.2, compute_fragment_metrics=True)
    except Exception as e:
        print(f"Failed on {output_csv}")
        print(e)

    try:
        track_from_nodes(output_csv, all_dir,
                         num_frames=num_frames, start_frame=0, frame_step=frame_step,
                         level='segment', voxel_size=0.2)
    except Exception as e:
        print(f"Failed on {output_csv}")
        print(e)

