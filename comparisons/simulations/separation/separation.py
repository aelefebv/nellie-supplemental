import numpy as np
import skimage

from utils import add_noise, save_ome_tif


def create_skeleton_sub_volume(length=50, separation_px=0):
    separation_px = int(np.round(separation_px))
    length = np.ceil(length).astype(int)
    if length == 0:
        length = 1
    assert length <= 50, "Length of the skeleton must be less than or equal to 50, and greater than 0."
    sub_volume = np.zeros((110, 110, 110), dtype=np.uint8)

    x = np.arange(length)
    y = np.ones_like(x) * 25
    z = np.ones_like(x) * 25

    for xi, yi, zi in zip(x, y, z):
        sub_volume[xi, yi, zi] = 1
        if separation_px > 0:
            sub_volume[xi, yi, zi+separation_px] = 1

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

    image_shape = (150, 150, 150)

    approx_diff_limit_um = 0.2  # um

    bit_depth_max = 65535
    intensity = bit_depth_max // 2  # keep it in the middle so no risk of over or under saturation with added noise

    step_size = 0.01
    start = 0.04
    stop = 0.3
    n = int((stop - start) / step_size) + 1
    px_size_vector = np.linspace(start, stop, n)

    step_size_um = 0.1
    step_start = 0.1
    step_stop = 1
    step_n = int((step_stop - step_start) / step_size_um) + 1
    separation_dist_vector_um = np.linspace(step_start, step_stop, step_n)
    for px_size_um in px_size_vector:
        separation_dist_vector = separation_dist_vector_um / px_size_um
        for separation_px in separation_dist_vector:
            min_radius_um = 0.2
            min_thickness_um = min_radius_um * 2

            # Total of 2 different thicknesses
            thickness_vector_um = np.linspace(min_thickness_um, 1, 2)
            thickness_vector_px = thickness_vector_um / px_size_um

            # Total of 2 different lengths
            stop_length = np.min([49 * px_size_um, 10])
            length_vector_um = np.linspace(min_thickness_um, stop_length, 2)
            length_vector_px = length_vector_um / px_size_um

            for thickness in thickness_vector_px:
                if thickness > separation_px*2 and thickness < separation_px*2 + 0.25:
                    continue
                for length in length_vector_px:
                    new_im = np.zeros(image_shape, dtype=np.uint16)

                    thickness_actual = np.min([thickness, length])
                    thickness_actual = np.max([thickness_actual, 1])
                    length_actual = np.max([thickness, length])
                    length_actual = np.max([length_actual, 1])

                    sub_volume = create_skeleton_sub_volume(length=length_actual, separation_px=separation_px)
                    sub_volume = skimage.morphology.binary_dilation(sub_volume, footprint=np.ones((3, 3, 3)))

                    sub_volume = skimage.morphology.skeletonize_3d(sub_volume)>0
                    sub_volume = populate_skeleton(sub_volume,
                                                   thickness=thickness_actual,
                                                   intensity=intensity)

                    new_im[20:new_im.shape[0]-20, 20:new_im.shape[1]-20, 20:new_im.shape[2]-20] = sub_volume

                    print(f'px_size_um: {px_size_um:.2f}, length_actual: {length_actual:.2f}, thickness_actual: {thickness_actual:.2f}, separation_px: {separation_px:.2f}')
                    save_name_no_ext = f'px_size_{px_size_um:.2f}-length_{length_actual:.2f}-thickness_{thickness_actual:.2f}-separation_{separation_px:.2f}'
                    save_name_no_ext = save_name_no_ext.replace('.', 'p')
                    save_name = f'{save_name_no_ext}.ome.tif'

                    no_noise_path = os.path.join(noiseless_path, f'no_noise-{save_name}')
                    dim_sizes = {'X': px_size_um, 'Y': px_size_um, 'Z': px_size_um}
                    save_ome_tif(no_noise_path, new_im, dim_sizes)

                    approx_diff_limit_px = approx_diff_limit_um / px_size_um
                    sigma_psf = approx_diff_limit_px / (2 * np.sqrt(2 * np.log(2)))
                    dim_sizes = {'X': px_size_um, 'Y': px_size_um, 'Z': px_size_um, 'T': 1}
                    for std in [128, 256, 512, 1024, 2048, 4096, 8192]:
                        noisy_im_series = []
                        for t in range(3):
                            noisy_im_series.append(add_noise(new_im, psf_sigma=sigma_psf, gaussian_mean=0, gaussian_std=std, apply_poisson=True))
                        im_name = f'std_{std}-{save_name}'
                        noisy_im_series = np.array(noisy_im_series)
                        path_im = os.path.join(output_path, im_name)
                        save_ome_tif(path_im, noisy_im_series, dim_sizes)
