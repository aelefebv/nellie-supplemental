import os
import pandas as pd
import tifffile
import numpy as np


def get_nellie_inputs(raw_basename, nellie_outputs, nellie_output_files):
    nellie_feature_branch_files = [
        os.path.join(nellie_outputs, file)
        for file in nellie_output_files
        if file.endswith("features_branches.csv") and raw_basename in file
    ]
    nellie_branch_tif_files = [
        os.path.join(nellie_outputs, file)
        for file in nellie_output_files
        if file.endswith("im_skel_relabelled.ome.tif") and raw_basename in file
    ]
    nellie_matched_files = (nellie_feature_branch_files[0], nellie_branch_tif_files[0])
    return nellie_matched_files


def get_nellie_outputs(csv, tif, time_bw_frames, px_size_um):
    # (csv, tif)
    nellie_df = pd.read_csv(csv)
    nellie_df_small = nellie_df[['t', 'reassigned_label_raw', 'label']]
    nellie_df_small_copy = nellie_df_small.copy()

    # ope the branch label ome tif
    nellie_label_tif = tifffile.imread(tif)
    all_label_vals = []
    remapped_vals = []
    centroids_by_label = {}
    label_persistence = {label: [] for label in nellie_df_small_copy['reassigned_label_raw'].unique()}
    for t in range(nellie_label_tif.shape[0]):
        label_coords = np.argwhere(nellie_label_tif[t])
        label_vals = nellie_label_tif[t, label_coords[:, 0], label_coords[:, 1], label_coords[:, 2]]
        all_label_vals.append(label_vals)
        # remap based on the reassigned labels
        frame_remap = nellie_df_small_copy.loc[nellie_df_small['t'] == t]

        remapped_vals_frame = [int(frame_remap.loc[frame_remap['label'] == label, 'reassigned_label_raw'].values[0]) for label in label_vals]
        # go through each unique remapped label and append the timepoint to the label_persistence dict
        for label in np.unique(remapped_vals_frame):
            label_persistence[label].append(t)
        remapped_vals.append(remapped_vals_frame)

        # for each unique label, get the centroid based on the coords
        centroid_remapped_label_tuple = [(label, np.mean(label_coords[remapped_vals[t] == label], axis=0)) for label in
                                         np.unique(remapped_vals[t])]
        # centroids_by_label.append(centroid_remapped_label_tuple)
        for label, centroid in centroid_remapped_label_tuple:
            if label not in centroids_by_label:
                centroids_by_label[label] = []
            centroids_by_label[label].append(centroid)

    # get the displacement between centroids with the same label between frames
    displacements = {}
    for label in centroids_by_label:
        centroids = centroids_by_label[label]
        for i in range(1, len(centroids)):
            prev_centroid = centroids[i - 1]
            curr_centroid = centroids[i]
            if i not in displacements:
                displacements[i] = []
            displacements[i].append(np.linalg.norm(curr_centroid - prev_centroid) * px_size_um / time_bw_frames)
    persistence_lengths = [len(times) for _, times in label_persistence.items()]

    return displacements, persistence_lengths


def get_mitotnt_seg_outputs(csv_path, time_bw_frames, fragments=True):
    try:
        track_df = pd.read_csv(csv_path)
    except:
        return {}, []
    tracks = {}
    # each row is a track, go through each row and add it to the array
    id_name = 'unique_frag_id' if fragments else 'unique_node_id'
    frame_name = 'frame' if fragments else 'frame_id'
    x_name = 'weighted_centroid.x' if fragments else 'x'
    y_name = 'weighted_centroid.y' if fragments else 'y'
    z_name = 'weighted_centroid.z' if fragments else 'z'
    persistances = {}
    for i, row in track_df.iterrows():
        if row[id_name] not in persistances:
            persistances[row[id_name]] = []
        persistances[row[id_name]].append(row[frame_name])
        if row[id_name] not in tracks:
            tracks[row[id_name]] = []
        tracks[row[id_name]].append((row[frame_name], [row[z_name], row[y_name], row[x_name]]))
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

def get_mitometer_inputs(raw_basename, mitometer_outputs):
    mitometer_files = os.listdir(mitometer_outputs)
    mitometer_files = [os.path.join(mitometer_outputs, file) for file in mitometer_files if raw_basename in file and file.endswith(".csv")]
    return mitometer_files[0]

def get_mitometer_outputs(csv_path, time_bw_frames):
    try:
        track_df = pd.read_csv(csv_path)
    except:
        return {}, []
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
    mean_displacements = {frame: np.nanmean(vals) for frame, vals in displacement_dict.items()}
    std_displacements = {frame: np.nanstd(vals) for frame, vals in displacement_dict.items()}
    n_displacements = {frame: len(vals) for frame, vals in displacement_dict.items()}
    q25_displacements = {frame: np.nanpercentile(vals, 25) for frame, vals in displacement_dict.items()}
    q50_displacements = {frame: np.nanpercentile(vals, 50) for frame, vals in displacement_dict.items()}
    q75_displacements = {frame: np.nanpercentile(vals, 75) for frame, vals in displacement_dict.items()}
    iqr_displacements = {frame: np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25) for frame, vals in displacement_dict.items()}
    # now average all the values
    mean_displacements = np.nanmean([val for val in mean_displacements.values()])
    std_displacements = np.nanmean([val for val in std_displacements.values()])
    n_displacements = np.nanmean([val for val in n_displacements.values()])
    q25_displacements = np.nanmean([val for val in q25_displacements.values()])
    q50_displacements = np.nanmean([val for val in q50_displacements.values()])
    q75_displacements = np.nanmean([val for val in q75_displacements.values()])
    iqr_displacements = np.nanmean([val for val in iqr_displacements.values()])
    return mean_displacements, std_displacements, n_displacements, q25_displacements, q50_displacements, q75_displacements, iqr_displacements


raw_path = "/Users/austin/GitHub/nellie-simulations/motion/angular/outputs"
save_path = os.path.join(raw_path, "linear_raw_outputs.csv")
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

    nellie_csv_path, nellie_tif_path = get_nellie_inputs(raw_basename, nellie_outputs, nellie_output_files)
    nellie_displacement, nellie_persistance = get_nellie_outputs(nellie_csv_path, nellie_tif_path, time_bw_frames, px_size_um)
    nellie_stats = get_stats(nellie_displacement)

    # mitotnt_csv = os.path.join(mitotnt_outputs, raw_basename.split('.')[0], 'tracking_outputs', 'final_node_tracks.csv')
    mitotnt_csv = os.path.join(mitotnt_outputs, raw_basename.split('.')[0] + 'fragment_tracks.csv')
    tnt_displacement, tnt_persistance = get_mitotnt_seg_outputs(mitotnt_csv, time_bw_frames, True)
    tnt_stats = get_stats(tnt_displacement)

    mitometer_path = get_mitometer_inputs(raw_basename.split('.ome')[0], mitometer_outputs)
    mm_displacement, mm_persistance = get_mitometer_outputs(mitometer_path, time_bw_frames)
    mm_stats = get_stats(mm_displacement)

    dict_to_save = {
        'std': std,
        'length': length,
        'time_bw_frames': time_bw_frames,
        'nellie_displacement_mean': nellie_stats[0],
        'nellie_displacement_std': nellie_stats[1],
        'nellie_n': nellie_stats[2],
        'nellie_displacement_q25': nellie_stats[3],
        'nellie_displacement_q50': nellie_stats[4],
        'nellie_displacement_q75': nellie_stats[5],
        'nellie_displacement_iqr': nellie_stats[6],

        'mitotnt_displacement_mean': tnt_stats[0],
        'mitotnt_displacement_std': tnt_stats[1],
        'mitotnt_n': tnt_stats[2],
        'mitotnt_displacement_q25': tnt_stats[3],
        'mitotnt_displacement_q50': tnt_stats[4],
        'mitotnt_displacement_q75': tnt_stats[5],
        'mitotnt_displacement_iqr': tnt_stats[6],

        'mitometer_displacement_mean': mm_stats[0],
        'mitometer_displacement_std': mm_stats[1],
        'mitometer_n': mm_stats[2],
        'mitometer_displacement_q25': mm_stats[3],
        'mitometer_displacement_q50': mm_stats[4],
        'mitometer_displacement_q75': mm_stats[5],
        'mitometer_displacement_iqr': mm_stats[6],

        'nellie_persistance_mean': np.nanmean(nellie_persistance),
        'nellie_persistance_std': np.nanstd(nellie_persistance),
        'nellie_persistance_q25': np.nanpercentile(nellie_persistance, 25),
        'nellie_persistance_q50': np.nanpercentile(nellie_persistance, 50),
        'nellie_persistance_q75': np.nanpercentile(nellie_persistance, 75),
        'nellie_persistance_iqr': np.nanpercentile(nellie_persistance, 75) - np.nanpercentile(nellie_persistance, 25),

        'mitotnt_persistance_mean': np.nanmean(tnt_persistance),
        'mitotnt_persistance_std': np.nanstd(tnt_persistance),
        'mitotnt_persistance_q25': np.nanpercentile(tnt_persistance, 25),
        'mitotnt_persistance_q50': np.nanpercentile(tnt_persistance, 50),
        'mitotnt_persistance_q75': np.nanpercentile(tnt_persistance, 75),
        'mitotnt_persistance_iqr': np.nanpercentile(tnt_persistance, 75) - np.nanpercentile(tnt_persistance, 25),

        'mitometer_persistance_mean': np.nanmean(mm_persistance),
        'mitometer_persistance_std': np.nanstd(mm_persistance),
        'mitometer_persistance_q25': np.nanpercentile(mm_persistance, 25),
        'mitometer_persistance_q50': np.nanpercentile(mm_persistance, 50),
        'mitometer_persistance_q75': np.nanpercentile(mm_persistance, 75),
        'mitometer_persistance_iqr': np.nanpercentile(mm_persistance, 75) - np.nanpercentile(mm_persistance, 25),

    }

    # save to csv, with headers if it doesn't exist
    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            f.write(','.join(dict_to_save.keys()) + '\n')
    with open(save_path, 'a') as f:
        f.write(','.join([str(val) for val in dict_to_save.values()]) + '\n')

