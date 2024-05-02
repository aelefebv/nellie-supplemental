import os
import pandas as pd
import tifffile
import numpy as np


def get_nellie_inputs(raw_basename, nellie_outputs, nellie_output_files):
    nellie_feature_branch_files = [
        os.path.join(nellie_outputs, file)
        for file in nellie_output_files
        if file.endswith("features_nodes.csv") and raw_basename in file
    ]
    return nellie_feature_branch_files[0]


def get_nellie_outputs(csv):
    nellie_df = pd.read_csv(csv)
    ave_angular_vel_mag = nellie_df[['t', 'ang_vel_mag_rel_mean']]
    # get the average for each frame
    ave_angular_vel_mag_frames = ave_angular_vel_mag.groupby('t').mean()
    ave_angular_vel_mag = ave_angular_vel_mag_frames.values
    # to degrees
    ave_angular_vel_mag = ave_angular_vel_mag * 180 / np.pi
    return ave_angular_vel_mag


def get_mitotnt_seg_outputs(csv_path, time_bw_frames, fragments=True):
    try:
        track_df = pd.read_csv(csv_path)
    except:
        return []
    tracks = {}
    # each row is a track, go through each row and add it to the array
    id_name = 'unique_frag_id' if fragments else 'unique_node_id'
    frame_name = 'frame' if fragments else 'frame_id'
    x_name = 'weighted_centroid.x' if fragments else 'x'
    y_name = 'weighted_centroid.y' if fragments else 'y'
    z_name = 'weighted_centroid.z' if fragments else 'z'
    for i, row in track_df.iterrows():
        if row[id_name] not in tracks:
            tracks[row[id_name]] = []
        tracks[row[id_name]].append((row[frame_name], [row[z_name], row[y_name], row[x_name]]))
    displacements = {}
    for track_num, track in tracks.items():
        for i in range(1, len(track)):
            prev_pos = track[i - 1][1]
            curr_pos = track[i][1]
            frame_diff = track[i][0] - track[i - 1][0]
            time_diff = frame_diff * time_bw_frames
            if track[i][0] not in displacements:
                displacements[track[i][0]] = []
            displacements[track[i][0]].append(np.linalg.norm(np.array(curr_pos) - np.array(prev_pos)) / time_diff)
    frame_displacements = []
    # get the average for each frame
    for frame, vals in displacements.items():
        frame_displacements.append(np.nanmean(vals))
    # to degrees
    all_vels = np.array(frame_displacements) * 180 / np.pi
    return all_vels

def get_mitometer_inputs(raw_basename, mitometer_outputs):
    mitometer_files = os.listdir(mitometer_outputs)
    mitometer_files = [os.path.join(mitometer_outputs, file) for file in mitometer_files if raw_basename in file and file.endswith(".csv")]
    return mitometer_files[0]

def get_mitometer_outputs(csv_path, time_bw_frames):
    try:
        track_df = pd.read_csv(csv_path)
    except:
        return []
    tracks = {}
    # each row is a track, go through each row and add it to the array
    persistances = {}
    for i, row in track_df.iterrows():
        track_num = int(row['track_num'])
        frame_num = int(row['frame'])
        if track_num not in persistances:
            persistances[track_num] = []
        persistances[track_num].append(frame_num)
        if track_num not in tracks:
            tracks[track_num] = []
        tracks[track_num].append((frame_num, [row['z'], row['y'], row['x']]))
    persistance_lengths = []
    for _, times in persistances.items():
        persistance_lengths.append(len(times))
    displacements = {}
    for track_num, track in tracks.items():
        for i in range(1, len(track)):
            prev_pos = track[i - 1][1]
            curr_pos = track[i][1]
            frame_diff = track[i][0] - track[i - 1][0]
            time_diff = frame_diff * time_bw_frames
            if track[i][0] not in displacements:
                displacements[track[i][0]] = []
            displacements[track[i][0]].append(np.linalg.norm(np.array(curr_pos) - np.array(prev_pos)) / time_diff)
    return displacements, persistance_lengths


def get_stats(displacement_dict):
    mean_displacements = np.nanmean(displacement_dict)
    std_displacements = np.nanstd(displacement_dict)
    n_displacements = len(displacement_dict)
    med_displacements = np.nanmedian(displacement_dict)
    perc_75 = np.nanpercentile(displacement_dict, 75)
    perc_25 = np.nanpercentile(displacement_dict, 25)
    return mean_displacements, std_displacements, n_displacements, med_displacements, perc_75, perc_25


raw_path = "/Users/austin/GitHub/nellie-simulations/motion/angular/outputs"
save_path = os.path.join(raw_path, "angular_raw_outputs.csv")
nellie_outputs = os.path.join(raw_path, "nellie_output")
mitotnt_outputs = os.path.join(raw_path, "mitotnt")
mitometer_outputs = os.path.join(raw_path, "mitometer")
noiseless_outputs = os.path.join(raw_path, "noiseless")

raw_files = os.listdir(raw_path)
raw_files = [os.path.join(raw_path, file) for file in raw_files if file.endswith(".tif")]
raw_basenames_no_ext = [os.path.basename(file).split(".tif")[0] for file in raw_files]

nellie_output_files = os.listdir(nellie_outputs)
mitotnt_output_files = os.listdir(mitotnt_outputs)
for raw_basename in raw_basenames_no_ext:
    length = int(raw_basename.split('-')[1].split('_')[-1])
    std = int(raw_basename.split('-')[2].split('_')[-1])
    time_str = raw_basename.split('.ome')[0].split('_')[-1]
    time_bw_frames = float(time_str.replace('p', '.'))
    px_size_um = 0.2

    for noiseless_file in os.listdir(noiseless_outputs):
        if time_str in noiseless_file.split('-')[-1]:
            noiseless_path = os.path.join(noiseless_outputs, noiseless_file)
            noiseless_tif = tifffile.imread(noiseless_path)
            break

    num_frames = noiseless_tif.shape[0] - 1

    nellie_csv_path = get_nellie_inputs(raw_basename, nellie_outputs, nellie_output_files)
    nellie_displacement = get_nellie_outputs(nellie_csv_path, time_bw_frames)
    nellie_stats = get_stats(nellie_displacement)

    mitotnt_csv = os.path.join(mitotnt_outputs, raw_basename.split('.')[0], 'tracking_outputs', 'final_node_tracks.csv')
    tnt_displacement = get_mitotnt_seg_outputs(mitotnt_csv, time_bw_frames, False)
    tnt_stats = get_stats(tnt_displacement)

    # mitometer can't do angular motion
    # mitometer_path = get_mitometer_inputs(raw_basename.split('.ome')[0], mitometer_outputs)
    # mm_displacement, mm_persistance = get_mitometer_outputs(mitometer_path, time_bw_frames)
    # mm_stats = get_stats(mm_displacement)

    dict_to_save = {
        'std': std,
        'length': length,
        'time_bw_frames': time_bw_frames,

        'nellie_displacement_mean': nellie_stats[0],
        'nellie_displacement_std': nellie_stats[1],
        'nellie_n': nellie_stats[2],
        'nellie_displacement_median': nellie_stats[3],
        'nellie_displacement_75': nellie_stats[4],
        'nellie_displacement_25': nellie_stats[5],

        'mitotnt_displacement_mean': tnt_stats[0],
        'mitotnt_displacement_std': tnt_stats[1],
        'mitotnt_n': tnt_stats[2],
        'mitotnt_displacement_median': tnt_stats[3],
        'mitotnt_displacement_75': tnt_stats[4],
        'mitotnt_displacement_25': tnt_stats[5],

        # 'mitometer_displacement_mean': mm_stats[0],
        # 'mitometer_displacement_std': mm_stats[1],
        # 'mitometer_n': mm_stats[2],
    }

    # save to csv, with headers if it doesn't exist
    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            f.write(','.join(dict_to_save.keys()) + '\n')
    with open(save_path, 'a') as f:
        f.write(','.join([str(val) for val in dict_to_save.values()]) + '\n')

