from scipy.optimize import linear_sum_assignment
from tifffile import tifffile
import ome_types
import numpy as np
from skimage import measure
import pandas as pd
from scipy.spatial import cKDTree

from comparisons.segmentation_comparisons import mitometer

xp = np
import scipy.ndimage as ndi
device_type = 'cpu'
import os
import time

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


def track(num_frames, mask_im, im, dim_sizes, weights, vel_thresh_um, frame_thresh):
    tracks = []

    frame_mito = {}
    for frame in range(num_frames):
        label_im, num_labels = ndi.label(mask_im[frame], structure=np.ones((3, 3, 3)))
        # if num_labels > 100:
        #     print(f'Frame {frame} has {num_labels} mitochondria. Skipping.')
        #     return []
        frame_regions = measure.regionprops(label_im, intensity_image=im[frame],
                                                spacing=(dim_sizes['Z'], dim_sizes['Y'], dim_sizes['X']))
        # remove any mito with only 1 voxel in any dimension
        frame_mito[frame] = [mito for mito in frame_regions if not np.any(np.array(mito.image.shape) == 1)]
        # get axis minor length
        for mito in frame_mito[frame]:
            try:
                mito.minor_axis_length
            except ValueError:
                # remove the mito
                frame_mito[frame].remove(mito)

        for mito in frame_mito[frame]:
            # get surface area
            v, f, _, _ = measure.marching_cubes(mito.intensity_image > 0,
                                                spacing=(dim_sizes['Z'], dim_sizes['Y'], dim_sizes['X']))
            mito.surface_area = measure.mesh_surface_area(v, f)
            mito.frame = frame
            if frame == 0:
                tracks.append({'mitos': [mito], 'frames': [frame], 'perfect': [True]})

    running_confidence_costs = []
    for frame in range(1, num_frames):
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
        distance_matrix_z = (distance_matrix - np.nanmean(distance_matrix)) / np.nanstd(distance_matrix) if np.nanstd(
            distance_matrix) != 0 else distance_matrix * 0
        volume_matrix_z = (volume_matrix - np.nanmean(volume_matrix)) / np.nanstd(volume_matrix) if np.nanstd(
            volume_matrix) != 0 else volume_matrix * 0
        majax_matrix_z = (majax_matrix - np.nanmean(majax_matrix)) / np.nanstd(majax_matrix) if np.nanstd(
            majax_matrix) != 0 else majax_matrix * 0
        minax_matrix_z = (minax_matrix - np.nanmean(minax_matrix)) / np.nanstd(minax_matrix) if np.nanstd(
            minax_matrix) != 0 else minax_matrix * 0
        z_axis_matrix_z = (z_axis_matrix - np.nanmean(z_axis_matrix)) / np.nanstd(z_axis_matrix) if np.nanstd(
            z_axis_matrix) != 0 else z_axis_matrix * 0
        solidity_matrix_z = (solidity_matrix - np.nanmean(solidity_matrix)) / np.nanstd(solidity_matrix) if np.nanstd(
            solidity_matrix) != 0 else solidity_matrix * 0
        surface_area_matrix_z = (surface_area_matrix - np.nanmean(surface_area_matrix)) / np.nanstd(
            surface_area_matrix) if np.nanstd(surface_area_matrix) != 0 else surface_area_matrix * 0
        intensity_matrix_z = (intensity_matrix - np.nanmean(intensity_matrix)) / np.nanstd(
            intensity_matrix) if np.nanstd(intensity_matrix) != 0 else intensity_matrix * 0

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
        # check where local and global assignments match
        confident_assignments = xp.where(min_local[row_ind] == col_ind)[0]

        # add matched mito from LAP to track
        for i, j in zip(row_ind, col_ind):
            confident = i in confident_assignments
            if j < len(tracks):
                tracks[j]['mitos'].append(frame_mito[frame][i])
                tracks[j]['frames'].append(frame)
                tracks[j]['perfect'].append(confident)
            else:
                tracks.append({'mitos': [frame_mito[frame][i]], 'frames': [frame], 'perfect': [False]})

        confident_costs = assign_matrix[row_ind[confident_assignments], col_ind[confident_assignments]]
        running_confidence_costs.extend(confident_costs)

    return tracks, frame_mito


def get_track_angles(tracks):
    tracks_angles = {'xy': [], 'xz': [], 'yz': []}
    for track in tracks:
        xy_angles = []
        xz_angles = []
        yz_angles = []
        for i in range(1, len(track['mitos'])):
            yz_angle = np.arctan2(track['mitos'][i].centroid[1] - track['mitos'][i - 1].centroid[1],
                                  track['mitos'][i].centroid[0] - track['mitos'][i - 1].centroid[0])
            xz_angle = np.arctan2(track['mitos'][i].centroid[2] - track['mitos'][i - 1].centroid[2],
                                  track['mitos'][i].centroid[0] - track['mitos'][i - 1].centroid[0])
            xy_angle = np.arctan2(track['mitos'][i].centroid[2] - track['mitos'][i - 1].centroid[2],
                                  track['mitos'][i].centroid[1] - track['mitos'][i - 1].centroid[1])

            # convert to degrees
            xy_angle = np.degrees(xy_angle)
            xz_angle = np.degrees(xz_angle)
            yz_angle = np.degrees(yz_angle)

            # constrain between 0 and 180
            xy_angle = np.abs(xy_angle) if xy_angle < 0 else 180 - xy_angle
            xz_angle = np.abs(xz_angle) if xz_angle < 0 else 180 - xz_angle
            yz_angle = np.abs(yz_angle) if yz_angle < 0 else 180 - yz_angle

            xy_angles.append(xy_angle)
            xz_angles.append(xz_angle)
            yz_angles.append(yz_angle)

        tracks_angles['xy'].append(xy_angles)
        tracks_angles['xz'].append(xz_angles)
        tracks_angles['yz'].append(yz_angles)

    return tracks_angles


def close_gaps(tracks, vel_thresh_um, frame_thresh, num_frames):
    num_tracks = len(tracks)
    final_tracks = tracks

    clean = False
    clean_num = 0
    while not clean:
        clean_num += 1

        lost_tracks = []
        new_tracks = []

        for track_num, track in enumerate(tracks):
            if track['frames'][-1] < num_frames - 1:
                lost_tracks.append(track_num)
            if track['frames'][0] > 0:
                new_tracks.append(track_num)

        lost_track_frames = np.array([tracks[lost_track]['frames'][-1] for lost_track in lost_tracks])
        new_track_frames = np.array([tracks[new_track]['frames'][0] for new_track in new_tracks])

        frame_diff_matrix = (new_track_frames[:, None] - lost_track_frames).astype(np.float64)
        frame_diff_matrix[frame_diff_matrix < 1] = frame_thresh + 1
        frame_diff_matrix[frame_diff_matrix > frame_thresh] = np.nan
        frame_diff_matrix[frame_diff_matrix > 0] = 1

        lost_track_coords = np.array([tracks[lost_track]['mitos'][-1].centroid_weighted for lost_track in lost_tracks])
        new_track_coords = np.array([tracks[new_track]['mitos'][0].centroid_weighted for new_track in new_tracks])

        if len(lost_track_coords) == 0 or len(new_track_coords) == 0:
            break

        # rows are new tracks, columns are lost tracks
        distance_matrix = np.linalg.norm(new_track_coords[:, None] - lost_track_coords, axis=-1)
        distance_matrix[distance_matrix > vel_thresh_um] = np.nan
        distance_matrix *= frame_diff_matrix

        valid_rows, valid_cols = np.where(np.isfinite(distance_matrix))
        valid_matches = list(zip(valid_rows, valid_cols))

        new_lost_cov_matrix = np.zeros((len(new_tracks), len(lost_tracks))) * np.nan
        single_match_cost = -100
        # if there's only a single non-nan value in the row of the distance matrix, then set the new_lost_cov_matrix value of that match index to -100
        for i in range(len(new_tracks)):
            if np.sum(np.isfinite(distance_matrix[i])) == 1:
                new_lost_cov_matrix[i, np.where(np.isfinite(distance_matrix[i]))[0][0]] = single_match_cost
        # get the new_lost_cov_matrix values of the valid matches
        match_vals = [new_lost_cov_matrix[match[0], match[1]] for match in valid_matches]
        # remove the matches from valid_matches
        valid_matches = [valid_matches[i] for i in range(len(valid_matches)) if match_vals[i] != single_match_cost]

        combined_tracks = []
        for matched_new, matched_lost in valid_matches:
            new_track = tracks[new_tracks[matched_new]]
            lost_track = tracks[lost_tracks[matched_lost]]
            combined_tracks.append({'mitos': lost_track['mitos'] + new_track['mitos'],
                                    'frames': lost_track['frames'] + new_track['frames'],
                                    'perfect': lost_track['perfect'] + new_track['perfect']})

        combined_track_angles = get_track_angles(combined_tracks)
        for match_num in range(len(valid_matches)):
            xy_CoV = np.std(combined_track_angles['xy'][match_num]) / np.mean(combined_track_angles['xy'][match_num])
            xz_CoV = np.std(combined_track_angles['xz'][match_num]) / np.mean(combined_track_angles['xz'][match_num])
            yz_CoV = np.std(combined_track_angles['yz'][match_num]) / np.mean(combined_track_angles['yz'][match_num])
            new_lost_cov_matrix[valid_matches[match_num]] = np.mean([xy_CoV, xz_CoV, yz_CoV])

        angle_CoV_threshold = 0.1
        new_lost_cov_matrix[new_lost_cov_matrix > angle_CoV_threshold] = np.nan
        new_lost_cov_matrix[np.isnan(new_lost_cov_matrix)] = angle_CoV_threshold + 1

        new_lost_matches = linear_sum_assignment(new_lost_cov_matrix)
        # remove matches with inf cost
        new_lost_matches = list(zip(new_lost_matches[0], new_lost_matches[1]))
        new_lost_matches = [match for match in new_lost_matches if new_lost_cov_matrix[match] < angle_CoV_threshold]
        # sort new_lost_matches by lowest to highest cost
        new_lost_matches = sorted(new_lost_matches, key=lambda x: new_lost_cov_matrix[x])

        combined_tracks = []
        remove_idxs = []
        for new_match, lost_match in new_lost_matches:
            if new_tracks[new_match] in remove_idxs or lost_tracks[lost_match] in remove_idxs:
                continue
            new_track = tracks[new_tracks[new_match]]
            lost_track = tracks[lost_tracks[lost_match]]
            combined_tracks.append({'mitos': lost_track['mitos'] + new_track['mitos'],
                                    'frames': lost_track['frames'] + new_track['frames'],
                                    'perfect': lost_track['perfect'] + new_track['perfect']})

            # remove the new_track and lost_track from the original tracks
            remove_idxs.append(new_tracks[new_match])
            remove_idxs.append(lost_tracks[lost_match])

        final_tracks = [track for idx, track in enumerate(tracks) if idx not in remove_idxs]

        final_tracks.extend(combined_tracks)
        num_new_tracks = len(final_tracks)
        print(f'Combined {len(combined_tracks)} tracks.')
        if len(combined_tracks) == 0:
            clean = True

        tracks = final_tracks

    return final_tracks


def clean_mask(mask_im, min_size):
    final_mask = []
    for mask_frame in mask_im:
        label_im, num_labels = ndi.label(mask_frame, structure=np.ones((3, 3, 3)))
        # use bincounts
        areas = np.bincount(label_im.ravel())[1:]
        mask_im = np.where(np.isin(label_im, np.where(areas > min_size)[0] + 1), label_im, 0) > 0
        final_mask.append(mask_im)
    final_mask = np.array(final_mask)
    return final_mask


def check_fission_volume(check_track_idx, all_tracks, std_range_fission, cv_perc_fission):
    num_tracks = len(all_tracks)

    pre_volume = np.ones((1, num_tracks)) * np.inf
    post_volume = np.ones((1, num_tracks)) * np.inf
    new_volume = np.ones((1, num_tracks)) * np.inf
    pre_std = np.ones((1, num_tracks)) * np.inf
    post_std = np.ones((1, num_tracks)) * np.inf
    new_std = np.ones((1, num_tracks)) * np.inf

    new_track = all_tracks[check_track_idx]
    new_track_volume = np.nanmean([new_track['mitos'][frame_num].area for frame_num in range(len(new_track['frames']))])
    new_track_std = np.nanstd([new_track['mitos'][frame_num].area for frame_num in range(len(new_track['frames']))])
    for track_num in range(num_tracks):
        if track_num == check_track_idx:
            continue

        other_track = all_tracks[track_num]
        frame_max = new_track['frames'][0]
        # ensure the other track has both frames before and during/after the new track
        if (np.sum(np.array(other_track['frames']) < frame_max) == 0) or (np.sum(np.array(other_track['frames']) >= frame_max) == 0):
            continue
        # get all indices of the frame before the new track
        frame_idxs_pre = np.where(np.array(other_track['frames']) < frame_max)[0]
        frame_idxs_post = np.where(np.array(other_track['frames']) >= frame_max)[0]
        prefission = [other_track['mitos'][frame_idx].area for frame_idx in frame_idxs_pre]
        postfission = [other_track['mitos'][frame_idx].area for frame_idx in frame_idxs_post]

        if len(prefission) < 2 or len(postfission) < 2:
            continue

        pre_volume[0, track_num] = np.nanmean(prefission)
        post_volume[0, track_num] = np.nanmean(postfission)
        new_volume[0, track_num] = new_track_volume

        pre_std[0, track_num] = np.nanstd(prefission)
        post_std[0, track_num] = np.nanstd(postfission)
        new_std[0, track_num] = new_track_std

    volume_diff = np.abs(pre_volume - (post_volume + new_volume))
    std_total = np.sqrt(pre_std**2 + post_std**2 + new_std**2)

    coeff_var_matrix_post = post_std / post_volume
    coeff_var_matrix_pre = pre_std / pre_volume
    coeff_var_matrix_new = new_std / new_volume

    coeff_var_matrix = coeff_var_matrix_post * coeff_var_matrix_pre * coeff_var_matrix_new
    # set nans to inf
    coeff_var_matrix[np.isnan(coeff_var_matrix)] = np.inf

    possible_fission_matrix = volume_diff < (std_range_fission * std_total)
    fission_matrix = possible_fission_matrix * (coeff_var_matrix < cv_perc_fission)

    fission_matrix[np.isnan(fission_matrix)] = False

    return fission_matrix

def check_fusion_volume(check_track_idx, all_tracks, std_range_fusion, cv_perc_fusion):
    num_tracks = len(all_tracks)

    pre_volume = np.ones((1, num_tracks)) * np.inf
    post_volume = np.ones((1, num_tracks)) * np.inf
    lost_volume = np.ones((1, num_tracks)) * np.inf
    pre_std = np.ones((1, num_tracks)) * np.inf
    post_std = np.ones((1, num_tracks)) * np.inf
    lost_std = np.ones((1, num_tracks)) * np.inf

    lost_track = all_tracks[check_track_idx]
    lost_track_volume = np.nanmean([lost_track['mitos'][frame_num].area for frame_num in range(len(lost_track['frames']))])
    lost_track_std = np.nanstd([lost_track['mitos'][frame_num].area for frame_num in range(len(lost_track['frames']))])
    for track_num in range(num_tracks):
        if track_num == check_track_idx:
            continue

        other_track = all_tracks[track_num]
        frame_min = lost_track['frames'][-1]
        # ensure the other track has both frames before and during/after the lost track
        if (np.sum(np.array(other_track['frames']) < frame_min) == 0) or (np.sum(np.array(other_track['frames']) >= frame_min) == 0):
            continue
        # get all indices of the frame before the new track
        frame_idxs_pre = np.where(np.array(other_track['frames']) < frame_min)[0]
        frame_idxs_post = np.where(np.array(other_track['frames']) >= frame_min)[0]
        prefusion = [other_track['mitos'][frame_idx].area for frame_idx in frame_idxs_pre]
        postfusion = [other_track['mitos'][frame_idx].area for frame_idx in frame_idxs_post]

        if len(prefusion) < 2 or len(postfusion) < 2:
            continue

        pre_volume[0, track_num] = np.nanmean(prefusion)
        post_volume[0, track_num] = np.nanmean(postfusion)
        lost_volume[0, track_num] = lost_track_volume

        pre_std[0, track_num] = np.nanstd(prefusion)
        post_std[0, track_num] = np.nanstd(postfusion)
        lost_std[0, track_num] = lost_track_std

    volume_diff = np.abs(post_volume - (pre_volume + lost_volume))
    std_total = np.sqrt(pre_std**2 + post_std**2 + lost_std**2)

    coeff_var_matrix_post = post_std / post_volume
    coeff_var_matrix_pre = pre_std / pre_volume
    coeff_var_matrix_lost = lost_std / lost_volume

    coeff_var_matrix = coeff_var_matrix_post * coeff_var_matrix_pre * coeff_var_matrix_lost
    # set nans to inf
    coeff_var_matrix[np.isnan(coeff_var_matrix)] = np.inf

    possible_fusion_matrix = volume_diff < (std_range_fusion * std_total)
    fusion_matrix = possible_fusion_matrix * (coeff_var_matrix < cv_perc_fusion)

    fusion_matrix[np.isnan(fusion_matrix)] = False

    return fusion_matrix


def count_fission_events(all_tracks, all_mito, dim_sizes, frame_thresh, dist_thresh_um):
    num_tracks = len(all_tracks)
    std_range_fission = np.inf
    cv_perc_fission = np.inf

    speed_thresh_um = dist_thresh_um * dim_sizes['T']

    num_mito_first_frame = len(all_mito[0])

    fission_track = np.zeros((num_tracks - num_mito_first_frame, num_tracks))
    extrema_fission = np.ones_like(fission_track) * np.inf
    fission_matrix = np.zeros_like(fission_track)
    extrema_fission_thresholded = np.ones_like(fission_track) * np.inf

    new_track_idx = [i for i in range(len(all_tracks)) if all_tracks[i]['frames'][0] > 0]

    for new_track_curr_idx, track_num in enumerate(new_track_idx):
        fission_track_first = np.zeros((frame_thresh, num_tracks))
        extrema_fission_first = np.ones_like(fission_track_first) * np.nan

        frame_difference = all_tracks[track_num]['frames'][0] - frame_thresh
        frame_check = range(np.max([0, frame_difference]), all_tracks[track_num]['frames'][0])

        tracks_to_check = [i for i in range(len(all_tracks)) if i != track_num]

        track_coords = np.array(all_tracks[track_num]['mitos'][0].coords_scaled)
        for frame_num in frame_check:
            for check_track_idx in tracks_to_check:
                check_track = all_tracks[check_track_idx]
                if (len(check_track['frames']) <= 2) or (frame_num not in check_track['frames']):
                    continue
                check_coords = np.array(check_track['mitos'][check_track['frames'].index(frame_num)].coords_scaled)
                # get closest distance between the two tracks
                tree = cKDTree(check_coords)
                dist, _ = tree.query(track_coords, k=1)
                min_dist = np.min(dist)
                if min_dist > speed_thresh_um:
                    continue
                extrema_fission_first[frame_num - frame_difference, check_track_idx] = min_dist
                fission_track_first[frame_num - frame_difference, check_track_idx] = 1

        fission_track[new_track_curr_idx, :] = np.sum(fission_track_first, axis=0)
        extrema_fission[new_track_curr_idx, :] = np.nanmean(extrema_fission_first, axis=0)
        # nans get converted to inf
        extrema_fission[np.isnan(extrema_fission)] = np.inf
        fission_matrix[new_track_curr_idx, :] = check_fission_volume(track_num, all_tracks, std_range_fission, cv_perc_fission)
        mask_fission = fission_matrix * (fission_track > 0)
        mask_fission[mask_fission==0] = np.inf

        extrema_fission_thresholded = extrema_fission * mask_fission
    # find the min value of the extrema_fission_thresholded for each new track
    min_extrema_fission = np.min(extrema_fission_thresholded, axis=1)
    num_events = np.sum(min_extrema_fission < np.inf)
    return num_events

def count_fusion_events(all_tracks, all_mito, dim_sizes, frame_thresh, dist_thresh_um, num_frames):
    num_tracks = len(all_tracks)
    std_range_fusion = np.inf
    cv_perc_fusion = np.inf

    speed_thresh_um = dist_thresh_um * dim_sizes['T']

    lost_tracks = []
    for track_num, track in enumerate(all_tracks):
        if track['frames'][-1] < num_frames - 1:
            lost_tracks.append(track_num)

    fusion_track = np.zeros((len(lost_tracks), num_tracks))
    extrema_fusion = np.ones_like(fusion_track) * np.inf
    fusion_matrix = np.zeros_like(fusion_track)
    extrema_fusion_thresholded = np.ones_like(fusion_track) * np.inf

    lost_track_idx = [i for i in range(len(all_tracks)) if i in lost_tracks]

    for lost_track_curr_idx, track_num in enumerate(lost_track_idx):
        fusion_track_first = np.zeros((frame_thresh, num_tracks))
        extrema_fusion_first = np.ones_like(fusion_track_first) * np.nan

        frame_difference = all_tracks[track_num]['frames'][-1] + frame_thresh
        frame_check = range(all_tracks[track_num]['frames'][-1], np.min([num_frames, frame_difference]))

        tracks_to_check = [i for i in range(len(all_tracks)) if i != track_num]

        track_coords = np.array(all_tracks[track_num]['mitos'][-1].coords_scaled)
        for frame_idx, frame_num in enumerate(frame_check):
            for check_track_idx in tracks_to_check:
                check_track = all_tracks[check_track_idx]
                if (len(check_track['frames']) <= 2) or (frame_num not in check_track['frames']):
                    continue
                check_coords = np.array(check_track['mitos'][check_track['frames'].index(frame_num)].coords_scaled)
                # get closest distance between the two tracks
                tree = cKDTree(check_coords)
                dist, _ = tree.query(track_coords, k=1)
                min_dist = np.min(dist)
                if min_dist > speed_thresh_um:
                    continue
                extrema_fusion_first[frame_idx, check_track_idx] = min_dist
                fusion_track_first[frame_idx, check_track_idx] = 1

        fusion_track[lost_track_curr_idx, :] = np.sum(fusion_track_first, axis=0)
        extrema_fusion[lost_track_curr_idx, :] = np.nanmean(extrema_fusion_first, axis=0)
        # nans get converted to inf
        extrema_fusion[np.isnan(extrema_fusion)] = np.inf
        fusion_matrix[lost_track_curr_idx, :] = check_fusion_volume(track_num, all_tracks, std_range_fusion, cv_perc_fusion)
        mask_fusion = fusion_matrix * (fusion_track > 0)
        mask_fusion[mask_fusion==0] = np.inf

        extrema_fusion_thresholded = extrema_fusion * mask_fusion
    # find the min value of the extrema_fission_thresholded for each new track
    min_extrema_fusion = np.min(extrema_fusion_thresholded, axis=1)
    num_events = np.sum(min_extrema_fusion < np.inf)
    return num_events


if __name__ == '__main__':
    distance_thresh_um = 3
    frame_thresh = 3
    visualize = False

    top_dir = '/Users/austin/test_files/timing_masks'
    all_files = os.listdir(top_dir)
    all_files.sort()
    all_files = [os.path.join(top_dir, file) for file in all_files if file.endswith('.tif')]

    raw_file_dir = '/Users/austin/test_files/time_stuff'
    raw_files = os.listdir(raw_file_dir)
    raw_files.sort()
    raw_files = [os.path.join(raw_file_dir, file) for file in raw_files if file.endswith('.tif')]

    matched_files = list(zip(all_files, raw_files))

    for mask_file, raw_file in matched_files:
        im_name = os.path.basename(raw_file)
        output_dir = os.path.join(top_dir, 'mitometer')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_name = os.path.join(output_dir, im_name)
        csv_path = os.path.join(output_dir, f'{im_name}-final_tracks.csv')
        mask_im = tifffile.imread(mask_file) > 0

        print(f'Running on {raw_file}')
        start_time = time.time()
        mask_im = clean_mask(mask_im, 4)

        raw_im = tifffile.imread(os.path.join(top_dir, mask_file))

        dim_sizes = {'X': 0.0655, 'Y': 0.0655,
                     'Z': 0.25, 'T': 4.536}

        vel_thresh_um = distance_thresh_um * dim_sizes['T']

        weights = {'vol': 1, 'majax': 1, 'minax': 1, 'z_axis': 1, 'solidity': 1, 'surface_area': 1, 'intensity': 1}
        num_frames = raw_im.shape[0]

        tracks, all_mito = track(num_frames, mask_im, raw_im, dim_sizes, weights, vel_thresh_um, frame_thresh)
        if len(tracks) > 1:
            final_tracks = close_gaps(tracks, vel_thresh_um, frame_thresh, num_frames)
        else:
            final_tracks = tracks
        track_info = []
        for track_num, final_track in enumerate(final_tracks):
            for mito in final_track['mitos']:
                info = {'track_num': track_num, 'frame': mito.frame, 'z': mito.centroid[0],
                        'y': mito.centroid[1], 'x': mito.centroid[2]}
                track_info.append(info)
        track_df = pd.DataFrame(track_info)
        track_df.to_csv(csv_path, index=False)

        dynamics_events_info = {'fission': count_fission_events(final_tracks, all_mito, dim_sizes, frame_thresh, distance_thresh_um),
                                'fusion': count_fusion_events(final_tracks, all_mito, dim_sizes, frame_thresh, distance_thresh_um, num_frames)}

        dynamics_events_df = pd.DataFrame(dynamics_events_info, index=[0])
        dynamics_events_df.to_csv(os.path.join(output_dir, f'{im_name}-dynamics_events.csv'), index=False)
        print(f'Finished {raw_file} in {time.time() - start_time} seconds.')