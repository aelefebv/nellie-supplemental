import os
import pandas as pd
import tifffile
import numpy as np


def get_nellie_inputs(raw_basename, nellie_outputs, nellie_output_files):
    nellie_feature_branch_files = [
        os.path.join(nellie_outputs, file)
        for file in nellie_output_files
        if file.endswith("features_components.csv") and raw_basename in file
    ]
    return nellie_feature_branch_files[0]


def get_nellie_outputs(csv):
    nellie_df = pd.read_csv(csv)
    nellie_df_small = nellie_df[['t', 'reassigned_label_raw', 'label']]
    max_frame_num = int(nellie_df_small['t'].max()) + 1
    total_num_reassigned_labels = len(nellie_df_small['reassigned_label_raw'].unique())

    events_per_frame = []
    label_differences = None
    for t in range(max_frame_num):
        num_unique_labels_in_t = len(nellie_df_small.loc[nellie_df_small['t'] == t, 'label'].unique())
        label_difference = num_unique_labels_in_t - total_num_reassigned_labels
        if label_differences is None:
            label_differences = [label_difference]
            events_per_frame.append(0)
            continue
        events_per_frame.append(label_difference - label_differences[-1])
        label_differences.append(label_difference)

    fission_events = sum([event for event in events_per_frame if event > 0])
    fusion_events = -sum([event for event in events_per_frame if event < 0])

    return fission_events, fusion_events


def get_mitotnt_seg_outputs(csv_path):
    try:
        track_df = pd.read_csv(csv_path)
    except:
        return 0, 0
    num_fission = 0
    num_fusion = 0
    # for each row, if the 'type' column is 'fission' or 'fusion', add to the count
    for i, row in track_df.iterrows():
        if row['type'] == 'fission':
            num_fission += 1
        elif row['type'] == 'fusion':
            num_fusion += 1
    return num_fission, num_fusion

def get_mitometer_inputs(raw_basename, mitometer_outputs):
    mitometer_files = os.listdir(mitometer_outputs)
    mitometer_files = [os.path.join(mitometer_outputs, file) for file in mitometer_files if raw_basename in file and file.endswith("dynamics_events.csv")]
    return mitometer_files[0]

def get_mitometer_outputs(csv_path):
    try:
        track_df = pd.read_csv(csv_path)
    except:
        return 0, 0
    # num fission is the first row value in the fission column
    num_fission = track_df['fission'].iloc[0]
    # num fusion is the first row value in the fusion column
    num_fusion = track_df['fusion'].iloc[0]
    return num_fission, num_fusion


def get_stats(displacement_dict):
    mean_displacements = {frame: np.nanmean(vals) for frame, vals in displacement_dict.items()}
    std_displacements = {frame: np.nanstd(vals) for frame, vals in displacement_dict.items()}
    n_displacements = {frame: len(vals) for frame, vals in displacement_dict.items()}
    q25_displacements = {frame: np.nanpercentile(vals, 25) for frame, vals in displacement_dict.items()}
    q50_displacements = {frame: np.nanpercentile(vals, 50) for frame, vals in displacement_dict.items()}
    q75_displacements = {frame: np.nanpercentile(vals, 75) for frame, vals in displacement_dict.items()}
    iqr_displacements = {frame: np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25) for frame, vals in displacement_dict.items()}
    center_frame = len(mean_displacements) // 2 + 1
    # middle frame is a transition and we don't have exact displacement values for it, so remove from quantification
    if center_frame in mean_displacements:
        del mean_displacements[center_frame]
        del std_displacements[center_frame]
        del n_displacements[center_frame]
        del q25_displacements[center_frame]
        del q50_displacements[center_frame]
        del q75_displacements[center_frame]
        del iqr_displacements[center_frame]
    # now average all the values
    mean_displacements = np.nanmean([val for val in mean_displacements.values()])
    std_displacements = np.nanmean([val for val in std_displacements.values()])
    n_displacements = np.nanmean([val for val in n_displacements.values()])
    q25_displacements = np.nanmean([val for val in q25_displacements.values()])
    q50_displacements = np.nanmean([val for val in q50_displacements.values()])
    q75_displacements = np.nanmean([val for val in q75_displacements.values()])
    iqr_displacements = np.nanmean([val for val in iqr_displacements.values()])
    return mean_displacements, std_displacements, n_displacements, q25_displacements, q50_displacements, q75_displacements, iqr_displacements


raw_path = "/Users/austin/GitHub/nellie-simulations/motion/fission_fusion/outputs"
save_path = os.path.join(raw_path, "fission_fusion_raw_outputs.csv")
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
    if '512' not in raw_basename:
        continue
    std = int(raw_basename.split('-')[2].split('_')[-1])
    fission_fusion = str(raw_basename.split('-')[1].split('_')[-1])
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
    nellie_fission, nellie_fusion = get_nellie_outputs(nellie_csv_path)

    mitotnt_csv = os.path.join(mitotnt_outputs, raw_basename.split('.')[0], 'tracking_outputs', 'remodeling_events.csv')
    tnt_fission, tnt_fusion = get_mitotnt_seg_outputs(mitotnt_csv)

    mitometer_path = get_mitometer_inputs(raw_basename.split('.ome')[0], mitometer_outputs)
    mm_fission, mm_fusion = get_mitometer_outputs(mitometer_path)

    dict_to_save = {
        'std': std,
        'long_axis': fission_fusion,
        'time_bw_frames': time_bw_frames,

        'nellie_fission': nellie_fission,
        'nellie_fusion': nellie_fusion,

        'mitotnt_fission': tnt_fission,
        'mitotnt_fusion': tnt_fusion,

        'mitometer_fission': mm_fission,
        'mitometer_fusion': mm_fusion,
    }

    # save to csv, with headers if it doesn't exist
    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            f.write(','.join(dict_to_save.keys()) + '\n')
    with open(save_path, 'a') as f:
        f.write(','.join([str(val) for val in dict_to_save.values()]) + '\n')

