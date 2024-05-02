import skimage
import numpy as np
import ome_types
from scipy.ndimage import gaussian_filter
from tifffile import tifffile
import os

from utils import save_ome_tif, add_noise


def create_skeleton_sub_volume(length=50, curvature_factor=5):
    length = np.ceil(length).astype(int)
    if length == 0:
        length = 1
    assert length <= 50, "Length of the skeleton must be less than or equal to 50, and greater than 0."
    sub_volume = np.zeros((50, 50, 50), dtype=np.uint8)

    x = np.arange(length)
    frequency = 2 * np.pi / length
    y = (25 + curvature_factor * np.sin(x * frequency)).astype(int)
    z = (25 + curvature_factor * np.cos(x * frequency)).astype(int)

    y = np.clip(y, 0, length-1)
    z = np.clip(z, 0, length-1)

    for xi, yi, zi in zip(x, y, z):
        sub_volume[xi, yi, zi] = 1

    return sub_volume


def populate_skeleton(sub_volume, thickness, intensity):
    radius = int(np.round(thickness / 2))
    skel_px = np.argwhere(sub_volume)
    new_sub_volume = np.zeros_like(sub_volume, dtype=np.uint16)
    for px in skel_px:
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

    image_shape = (510, 510, 510)
    large_image = np.zeros(image_shape, dtype=np.uint16)

    grid_size = (10, 10, 10)  # 10x10x10 grid of sub-volumes

    total_num_sub_volumes = grid_size[0] * grid_size[1] * grid_size[2]

    px_size_um = 0.1

    min_radius_um = 0.2
    min_length_um = min_radius_um * 2
    start_length_px = min_length_um / px_size_um

    length_vector = np.linspace(start_length_px, 40, grid_size[0])

    thickness_vector_um = np.linspace(min_length_um, 1, grid_size[0])
    thickness_vector_px = thickness_vector_um / px_size_um

    # from 10% to 90%
    intensity_vector = np.linspace(6553, 58982, grid_size[0])

    current_thickness = 0
    for ix in range(grid_size[0]):
        current_length = 0
        for iy in range(grid_size[2]):
            current_intensity = 0
            for iz in range(grid_size[1]):
                thickness = thickness_vector_px[current_thickness]
                length = length_vector[current_length]
                intensity = intensity_vector[current_intensity]

                length = np.max([length, thickness])
                sub_volume = create_skeleton_sub_volume(length=length)
                sub_volume = skimage.morphology.binary_dilation(sub_volume, footprint=np.ones((3, 3, 3)))

                sub_volume = skimage.morphology.skeletonize_3d(sub_volume)>0
                sub_volume = populate_skeleton(sub_volume,
                                               thickness=thickness,
                                               intensity=intensity)

                start_x, start_y, start_z = ix * 50 + 5, iy * 50 + 5, iz * 50 + 5

                large_image[start_x:start_x + 50, start_y:start_y + 50, start_z:start_z + 50] = sub_volume

                current_intensity += 1
            current_length += 1
        current_thickness += 1


    dim_sizes = {'X': px_size_um, 'Y': px_size_um, 'Z': px_size_um}
    noiseless_name = 'no_noise.ome.tif'
    save_ome_tif(os.path.join(noiseless_path, noiseless_name), large_image, dim_sizes)

    approx_diff_limit_um = 0.2  # um
    approx_diff_limit_px = approx_diff_limit_um / px_size_um
    sigma_psf = approx_diff_limit_px / (2 * np.sqrt(2 * np.log(2)))

    dim_sizes = {'X': px_size_um, 'Y': px_size_um, 'Z': px_size_um, 'T': 1}
    std_vector = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    for i, std in enumerate(std_vector):
        print(f'Processing image {i + 1} of {len(std_vector)}')
        noisy_im_series = []
        for t in range(3):
            noisy_im_series.append(add_noise(large_image, psf_sigma=sigma_psf, gaussian_mean=0, gaussian_std=std, apply_poisson=True))
        noisy_im_series = np.array(noisy_im_series)
        path_im = os.path.join(output_path, f'gaussian_std_{std}.ome.tif')
        save_ome_tif(path_im, noisy_im_series, dim_sizes)

