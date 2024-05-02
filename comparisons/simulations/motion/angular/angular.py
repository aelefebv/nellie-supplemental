import numpy as np
import ome_types
from scipy.ndimage import rotate
import skimage.morphology

from utils import add_noise, save_ome_tif


def create_cylinder(grid_size, center, radius, length, axis='z'):
    Z, Y, X = np.ogrid[:grid_size, :grid_size, :grid_size]
    if axis == 'z':
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        within_radius = dist_from_center <= radius
        within_length = (Z >= center[2] - length / 2) & (Z <= center[2] + length / 2)
    elif axis == 'y':
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Z - center[2]) ** 2)
        within_radius = dist_from_center <= radius
        within_length = (Y >= center[1] - length / 2) & (Y <= center[1] + length / 2)
    elif axis == 'x':
        dist_from_center = np.sqrt((Y - center[1]) ** 2 + (Z - center[2]) ** 2)
        within_radius = dist_from_center <= radius
        within_length = (X >= center[0] - length / 2) & (X <= center[0] + length / 2)

    cylinder = within_radius & within_length
    return cylinder.astype(np.uint8)


def generate_rotation_frames(object, num_frames, axis='z'):
    frames = []
    for frame in range(num_frames):
        angle = 5 * frame  # 5 degrees per frame (10 degrees/second)
        rotated_object = rotate(object, angle, axes=(1, 0), reshape=False, order=0) if axis == 'z' \
            else rotate(object, angle, axes=(2, 1), reshape=False, order=0) if axis == 'x' \
            else rotate(object, angle, axes=(0, 2), reshape=False, order=0)
        frames.append(rotated_object)
    return frames


def populate_skeleton(sub_volume, thickness, intensity):
    skel_px = np.argwhere(sub_volume)
    new_sub_volume = np.zeros_like(sub_volume, dtype=np.uint16)
    for px in skel_px:
        radius = int(np.round(thickness / 2))
        # radius += np.random.choice([-1, 0, 1])
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

    return new_sub_volume


if __name__ == "__main__":
    import os
    output_path = 'outputs'

    noiseless_path = os.path.join(output_path, 'noiseless')
    if not os.path.exists(noiseless_path):
        os.makedirs(noiseless_path)

    # Configuration
    grid_size = 50
    center = [grid_size // 2, grid_size // 2, grid_size // 2]
    thickness = 5
    intensity = 25000
    num_frames = 33

    lengths = [32]  # Length of the cylinder
    std_vals = [512]
    t_steps = [0.5, 1, 2, 4, 8]

    for length in lengths:
        # Create the initial cylinder
        cylinder = create_cylinder(grid_size, center, 5, length+1, axis='z')  # Cylinder along z-axis
        populated_frame = populate_skeleton(skimage.morphology.skeletonize(cylinder), intensity=intensity, thickness=thickness)

        # Generate rotation frames
        frames = np.array(generate_rotation_frames(populated_frame, num_frames, axis='y'), dtype=np.uint16)

        approx_diff_limit_um = 0.2  # um
        px_size_um = 0.2

        approx_diff_limit_px = approx_diff_limit_um / px_size_um
        sigma_psf = approx_diff_limit_px / (2 * np.sqrt(2 * np.log(2)))

        for std_num, std in enumerate(std_vals):
            noisy = add_noise(frames, psf_sigma=sigma_psf, gaussian_std=std, apply_poisson=True)

            for t in t_steps:
                dim_sizes = {'X': px_size_um, 'Y': px_size_um, 'Z': px_size_um, 'T': t}
                t_str = str(t).replace('.', 'p')

                new_im = noisy[::int(t*2)]
                path_im = os.path.join(output_path, f"angular-length_{int(length)}-std_{int(std)}-t_{t_str}.ome.tif")
                save_ome_tif(path_im, new_im, dim_sizes)

                if std_num == 0:
                    new_noiseless_im = frames[::int(t*2)]
                    noiseless_path_im = f"no_noise-angular-length_{int(length)}-t_{t_str}.ome.tif"
                    save_ome_tif(os.path.join(noiseless_path, noiseless_path_im), new_noiseless_im, dim_sizes)


