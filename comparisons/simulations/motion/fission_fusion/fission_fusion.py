import numpy as np
from scipy.ndimage import gaussian_filter

from utils import add_noise, save_ome_tif


def generate_initial_lines(grid_size, central_point, line_lengths):
    frame = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    branches = []
    for axis, length in enumerate(line_lengths):
        line = np.array([central_point.copy() for _ in range(length)])
        for i in range(2):
            line_copy = line.copy()
            if i == 0:
                line_copy[:, axis] += np.arange(length)  # Extend line along the axis
            else:
                line_copy[:, axis] -= np.arange(length)
            branches.append(line_copy)
            for point in line_copy:
                if np.all((point >= 0) & (point < grid_size)):
                    frame[tuple(point)] = 1
    return frame, branches


def generate_frames(branches, num_frames, grid_size):
    frames = []
    move_axes = [0, 0, 1, 1, 2, 2]
    move_dir = [1, -1, 1, -1, 1, -1]
    # Each branch should move along its own axis, but delayed by 3 frames
    for frame_num in range(num_frames):
        frame = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
        for branch_num, branch in enumerate(branches):
            if frame_num > branch_num*3:
                branch = branch + (frame_num - (branch_num*3) + 1) * np.array([move_dir[branch_num] if i == move_axes[branch_num]
                                                              else 0 for i in range(3)])
            for point in branch:
                if np.all((point >= 0) & (point < grid_size)):
                    frame[tuple(point)] = branch_num + 1
        frames.append(frame)
    return frames


def populate_skeleton(sub_volume, thickness, intensity):
    new_sub_volumes = []
    for label in np.unique(sub_volume):
        if label == 0:
            continue

        label_im = sub_volume == label
        temp_vol = []
        for temporal_frame in label_im:
            skel_px = np.argwhere(temporal_frame)
            new_sub_volume = np.zeros_like(temporal_frame, dtype=np.uint16)
            for px in skel_px:
                radius = int(np.round(thickness / 2)) - 1
                radius = np.max([2, radius])
                center_px = px
                x_min = max(0, int(np.round(center_px[2] - radius)))
                x_max = min(new_sub_volume.shape[2], int(np.round(center_px[2] + radius))+1)
                x_len = x_max - x_min
                x_lims = (x_min, x_max)

                y_min = max(0, int(np.round(center_px[1] - radius)))
                y_max = min(new_sub_volume.shape[1], int(np.round(center_px[1] + radius))+1)
                y_len = y_max - y_min
                y_lims = (y_min, y_max)

                z_min = max(0, int(np.round(center_px[0] - radius)))
                z_max = min(new_sub_volume.shape[0], int(np.round(center_px[0] + radius))+1)
                z_len = z_max - z_min
                z_lims = (z_min, z_max)

                sphere = np.zeros((z_len, y_len, x_len), dtype=np.uint8)
                for z in range(z_len):
                    for y in range(y_len):
                        for x in range(x_len):
                            if (z - radius) ** 2 + (y - radius) ** 2 + (x - radius) ** 2 <= radius ** 2:
                                sphere[z, y, x] = 1

                new_sub_volume[z_lims[0]:z_lims[0] + z_len, y_lims[0]:y_lims[0] + y_len, x_lims[0]:x_lims[0] + x_len] += sphere  # current_vals
            temp_vol.append(new_sub_volume)
        new_sub_volumes.append(np.array(temp_vol))
    # add all new_sub_volumes together
    new_sub_volumes = (np.sum(new_sub_volumes, axis=0) > 0) * intensity
    new_sub_volumes = gaussian_filter(new_sub_volumes, 1)
    return new_sub_volumes


if __name__ == "__main__":
    import os
    output_path = 'outputs'

    noiseless_path = os.path.join(output_path, 'noiseless')
    if not os.path.exists(noiseless_path):
        os.makedirs(noiseless_path)

    grid_size = 100
    line_lengths = [5, 8, 10]  # Length of each line along x, y, z
    central_point = np.array([grid_size // 2, grid_size // 2, grid_size // 2])
    intensity = 25000
    thickness = 5
    num_frames = 33

    std_vals = [512]
    t_steps = [0.5, 1, 2, 4, 8]

    initial_frame, branches = generate_initial_lines(grid_size, central_point, line_lengths)

    moving_frames = np.array(generate_frames(branches, num_frames, grid_size))

    populated = populate_skeleton(moving_frames, thickness, intensity)

    approx_diff_limit_um = 0.2  # um
    px_size_um = 0.2

    approx_diff_limit_px = approx_diff_limit_um / px_size_um
    sigma_psf = approx_diff_limit_px / (2 * np.sqrt(2 * np.log(2)))

    for std_num, std in enumerate(std_vals):
        noisy = add_noise(populated, psf_sigma=sigma_psf, gaussian_std=std, apply_poisson=True)

        for t in t_steps:
            dim_sizes = {'X': px_size_um, 'Y': px_size_um, 'Z': px_size_um, 'T': t}
            t_str = str(t).replace('.', 'p')

            new_im = noisy[::int(t * 2)]
            fission_name = f"fission-std_{int(std)}-t_{t_str}.ome.tif"
            fission_path_im = os.path.join(output_path, f"fission-std_{fission_name}")
            save_ome_tif(fission_path_im, new_im, dim_sizes)
            if std_num == 0:
                noiseless_new_im = populated[::int(t * 2)]
                save_ome_tif(os.path.join(noiseless_path, f"no_noise-{fission_name}"), noiseless_new_im, dim_sizes)

            reversed_new_im = new_im[::-1]
            fusion_name = f"fusion-std_{int(std)}-t_{t_str}.ome.tif"
            fusion_path_im = os.path.join(output_path, f"fusion-std_{fusion_name}")
            save_ome_tif(fusion_path_im, reversed_new_im, dim_sizes)
            if std_num == 0:
                noiseless_reversed_new_im = populated[::-1][::int(t * 2)]
                save_ome_tif(os.path.join(noiseless_path, f"no_noise-{fusion_name}"), noiseless_reversed_new_im, dim_sizes)

