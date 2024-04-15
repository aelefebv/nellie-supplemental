import pandas as pd
import os
import ome_types
import tifffile
import numpy as np


im_path = '/Users/austin/GitHub/nellie-simulations/motion/angular/outputs/angular-length_32-std_512-t_1.ome.tif'

im_dir, im_name = os.path.dirname(im_path), os.path.basename(im_path).split('.')[0]
output_path = f'{im_dir}/mitoTNT/{im_name}/tracking_outputs'
track_path = os.path.join(output_path, 'final_node_tracks.csv')
df = pd.read_csv(track_path)

# get metadata
ome_xml = tifffile.tiffcomment(im_path)
ome = ome_types.from_xml(ome_xml)
frame_interval = ome.images[0].pixels.time_increment


class Track:
    def __init__(self):
        self.node_ids = []
        self.fragment_ids = []
        self.segment_ids = []
        self.x = []
        self.y = []
        self.z = []
        self.speeds = []
        self.frames = []

    def add_node(self, node_id, fragment_id, segment_id, x, y, z, frame):
        self.node_ids.append(node_id)
        self.fragment_ids.append(fragment_id)
        self.segment_ids.append(segment_id)
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)
        self.frames.append(frame)

    def get_speed(self, time_between_frames):
        x = np.array(self.x)
        y = np.array(self.y)
        z = np.array(self.z)
        frame_differences = np.array(self.frames[1:]) - np.array(self.frames[:-1])
        distances = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2 + (z[1:] - z[:-1])**2)
        speeds = distances / (frame_differences * time_between_frames)
        self.speeds = speeds
        return speeds


node_ids = df['unique_node_id'].unique()
tracks = []
for node in node_ids:
    track = Track()
    node_df = df[df['unique_node_id'] == node]
    for i, row in node_df.iterrows():
        track.add_node(row['unique_node_id'], row['frame_frag_id'], row['frame_seg_id'], row['x'], row['y'], row['z'], row['frame_id'])
    tracks.append(track)

for track in tracks:
    track.get_speed(frame_interval)

fragment_speeds = {}
for track in tracks:
    for i, fragment_id in enumerate(track.fragment_ids[:-1]):
        if fragment_id not in fragment_speeds.keys():
            fragment_speeds[fragment_id] = {}
        if track.frames[i] not in fragment_speeds[fragment_id].keys():
            fragment_speeds[fragment_id][track.frames[i]] = []
        fragment_speeds[fragment_id][track.frames[i]].append(track.speeds[i])

segment_speeds = {}
for track in tracks:
    for i, segment_id in enumerate(track.segment_ids[:-1]):
        if segment_id not in segment_speeds.keys():
            segment_speeds[segment_id] = {}
        if track.frames[i] not in segment_speeds[segment_id].keys():
            segment_speeds[segment_id][track.frames[i]] = []
        segment_speeds[segment_id][track.frames[i]].append(track.speeds[i])

# save track speeds, segment speeds, and fragment speeds into separate csvs
# rows are individual tracks/fragments/segments, columns are frame numbers
largest_frame_num = max([max(track.frames) for track in tracks])

track_csv_path = os.path.join(output_path, 'node_speeds.csv')
segment_csv_path = os.path.join(output_path, 'object_speeds.csv')
fragment_csv_path = os.path.join(output_path, 'branch_speeds.csv')

track_speeds_out = np.zeros((len(tracks), largest_frame_num))
segment_speeds_out = np.zeros((len(segment_speeds), largest_frame_num))
fragment_speeds_out = np.zeros((len(fragment_speeds), largest_frame_num))

for i, track in enumerate(tracks):
    for j, frame in enumerate(track.frames[:-1]):
        track_speeds_out[i, frame] = track.speeds[j]

for i, segment_id in enumerate(segment_speeds.keys()):
    for j, frame in enumerate(segment_speeds[segment_id].keys()):
        segment_speeds_out[i, frame] = np.mean(segment_speeds[segment_id][frame])

for i, fragment_id in enumerate(fragment_speeds.keys()):
    for j, frame in enumerate(fragment_speeds[fragment_id].keys()):
        fragment_speeds_out[i, frame] = np.mean(fragment_speeds[fragment_id][frame])

pd.DataFrame(track_speeds_out).to_csv(track_csv_path)
pd.DataFrame(segment_speeds_out).to_csv(segment_csv_path)
pd.DataFrame(fragment_speeds_out).to_csv(fragment_csv_path)
