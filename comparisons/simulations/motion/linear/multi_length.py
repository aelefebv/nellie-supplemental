import numpy as np
from scipy.ndimage import gaussian_filter

from utils import add_noise, save_ome_tif


def generate_initial_lines(grid_size, central_point, line_lengths):
    frame = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    branches = []
    for axis, length in enumerate(line_lengths):
        line = np.array([central_point.copy() for _ in range(length)])
        line[:, axis] += np.arange(length)  # Extend line along the axis
        branches.append(line)
        for point in line:
            if np.all((point >= 0) & (point < grid_size)):
                frame[tuple(point)] = 1
    return frame, branches

def generate_frames(branches, num_frames, grid_size, move_along_longest_axis=False):
    frames = []
    for frame_num in range(num_frames):
        frame = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
        for branch_num, branch in enumerate(branches):
            if move_along_longest_axis:
                # let's just make longest axis one of the non-branch_num axes, so just the branch_num axis +1
                long_axis = branch_num + 1
                if long_axis == 3:
                    long_axis = 0
                moved_branch = branch + (frame_num + 1) * np.array([1 if i == long_axis else 0 for i in range(3)])
            else:
                moved_branch = branch + (frame_num + 1) * np.array([1 if i == branch_num else 0 for i in range(3)])
            for point in moved_branch:
                if np.all((point >= 0) & (point < grid_size)):
                    frame[tuple(point)] = branch_num + 1
        frames.append(frame)
    return frames


def populate_skeleton(sub_volume, thicknesses, intensities):
    new_sub_volumes = []
    for label in np.unique(sub_volume):
        if label == 0:
            continue
        thickness = thicknesses[label - 1]
        intensity = intensities[label - 1]

        label_im = sub_volume == label
        temp_vol = []
        for temporal_frame in label_im:
            skel_px = np.argwhere(temporal_frame)
            new_sub_volume = np.zeros_like(temporal_frame, dtype=np.uint16)
            for px in skel_px:
                # radius = int(np.round(thickness / 2))
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
            new_sub_volume = (new_sub_volume>0) * intensity
            new_sub_volume = gaussian_filter(new_sub_volume, 1)
            temp_vol.append(new_sub_volume)
        new_sub_volumes.append(np.array(temp_vol))
    # add all new_sub_volumes together
    new_sub_volumes = np.sum(new_sub_volumes, axis=0)
    return new_sub_volumes


if __name__ == "__main__":
    import os
    output_path = 'outputs'

    noiseless_path = os.path.join(output_path, 'noiseless')
    if not os.path.exists(noiseless_path):
        os.makedirs(noiseless_path)

    line_lengths = [20, 10, 6]  # Length of each line along x, y, z
    thicknesses = [5, 8, 5]
    intensities = [20000, 25000, 40000]
    grid_size = np.max(line_lengths) * 4
    central_point = np.array([grid_size // 2, grid_size // 2, grid_size // 2]) - np.max(line_lengths) // 4 * 3
    num_frames = 33

    std_vals = [512]
    t_steps = [0.5, 1, 2, 4, 8]

    initial_frame, branches = generate_initial_lines(grid_size, central_point, line_lengths)

    for long_axis in [True, False]:
        if long_axis:
            start_frame = 7  # otherwise objects overlap due to orientation
        else:
            start_frame = 5
        moving_frames = generate_frames(branches, num_frames, grid_size, move_along_longest_axis=long_axis)[start_frame:]
        reversed = moving_frames[::-1]

        full_frames = np.concatenate((reversed, moving_frames), axis=0)

        populated = np.array(populate_skeleton(full_frames, thicknesses, intensities), dtype=np.uint16)

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
                path_im = os.path.join(output_path, f"multi_length-long_axis_{long_axis}-std_{int(std)}-t_{t_str}.ome.tif")
                save_ome_tif(path_im, new_im, dim_sizes)

                if std_num == 0:
                    new_noiseless_im = populated[::int(t * 2)]
                    noiseless_path_im = f"no_noise-multi_length-long_axis_{long_axis}-t_{t_str}.ome.tif"
                    save_ome_tif(os.path.join(noiseless_path, noiseless_path_im), new_noiseless_im, dim_sizes)
