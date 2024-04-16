import numpy as np
import tifffile
import os

try:
    import cupy as xp
    import cupyx.scipy.ndimage as ndi
    device_type = 'cuda'
except ImportError:
    xp = np
    import scipy.ndimage as ndi
    device_type = 'cpu'

import skimage

def remove_background(stack):
    # as defined in paper
    min_rad_um = 0.3
    max_rad_um = 1

    px_size = 0.1
    min_rad_px = min_rad_um / px_size
    max_rad_px = max_rad_um / px_size

    im_min_med = xp.zeros_like(stack)
    im_bg_removed = xp.zeros_like(stack)
    min_rad_px = np.max([2, min_rad_px])
    for z_slice in range(stack.shape[0]):
        im_med_filtered = xp.zeros((round(max_rad_px) - round(min_rad_px) + 1, stack.shape[1], stack.shape[2]), dtype=stack.dtype)
        im_med_filtered[0] = stack[z_slice]

        circle_filt_num = 0
        for circle_filt in range(round(min_rad_px), round(max_rad_px) + 1):
            y, x = xp.ogrid[-circle_filt:circle_filt + 1, -circle_filt:circle_filt + 1]
            mask = x ** 2 + y ** 2 <= circle_filt ** 2
            disk_filter = xp.array(mask, dtype=xp.uint8)

            im_med_filtered[circle_filt_num] = ndi.median_filter(stack[z_slice], footprint=disk_filter)

            circle_filt_num += 1

        im_min_med[z_slice] = xp.min(im_med_filtered, axis=0)
        # anywhere im_min_med is greater than stack, set to stack
        im_min_med[z_slice] = xp.where(im_min_med[z_slice] > stack[z_slice], stack[z_slice], im_min_med[z_slice])
        im_bg_removed[z_slice] = stack[z_slice] - im_min_med[z_slice]

    if device_type == 'cuda':
        im_bg_removed = im_bg_removed.get()
    return im_bg_removed


def add_noise(stack):
    noisy_image = stack.copy().astype(xp.float32)
    noisy_image = xp.clip(noisy_image, 0, xp.max(stack))
    noisy_image = xp.random.poisson(noisy_image).astype(xp.float32)
    noisy_image = xp.clip(noisy_image, 0, 255)
    # convert back to original dtype
    noisy_image = noisy_image.astype(stack.dtype)
    return noisy_image


def run(im_path):
    top_dir = os.path.dirname(im_path)
    im_name = os.path.basename(im_path)

    output_dir = os.path.join(top_dir, 'mitometer')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # all_files = os.listdir(time_series_dir)
    # file_names = [file for file in all_files if file.endswith('.tif')]
    # final_seg = None
    # for file_num, tif_file in enumerate(file_names):
    output_name = os.path.join(output_dir, im_name)
    if os.path.exists(output_name):
        print(f'already exists, skipping')
        # open im, return it
        final_seg = tifffile.imread(output_name)
        return final_seg

    stack = tifffile.imread(im_path)
    num_t = stack.shape[0]
    check_num_t = min(10, num_t)

    # def run_frame(stack):
    stack = xp.asarray(stack)
    # convert 16-bit to 8-bit image as done in paper
    print(stack.dtype)
    if stack.dtype == 'uint16':
        stack = xp.round(stack / 256).astype(xp.uint8)

    all_im_bg_removed_timepoints = []
    for t in range(num_t):
        print(f'Removing background from timepoint {t}.')
        all_im_bg_removed_timepoints.append(remove_background(stack[t]))
    all_im_bg_removed_timepoints = xp.asarray(all_im_bg_removed_timepoints)

    sigma_matrix = xp.arange(0.33, 0.5, 0.01)
    if xp.max(all_im_bg_removed_timepoints) == 0:
        intensity_quantile_80 = 0
    else:
        intensity_quantile_80 = xp.quantile(all_im_bg_removed_timepoints[all_im_bg_removed_timepoints > 0], 0.8)
    # if not intensity_quantile_80:
    print(f'{intensity_quantile_80=}')
    thresh_matrix = range(2, int(intensity_quantile_80))
    if intensity_quantile_80 <= 2:
        thresh_matrix = [2]
    num_components = xp.zeros((len(all_im_bg_removed_timepoints), len(thresh_matrix), len(sigma_matrix)))
    median_area = xp.zeros((len(all_im_bg_removed_timepoints), len(thresh_matrix), len(sigma_matrix)))
    for sigma_num, sigma in enumerate(sigma_matrix):
        im_filtered = xp.zeros_like(all_im_bg_removed_timepoints)
        for t in range(check_num_t):
            im_filtered[t] = ndi.gaussian_filter(all_im_bg_removed_timepoints[t], sigma=sigma)
        for thresh_num, thresh in enumerate(thresh_matrix):
            print(f'Running connected components for {thresh=}, {sigma=}.')
            for t in range(check_num_t):
                im = im_filtered[t] > thresh
                labeled_im, num_labels = ndi.label(im)
                num_components[t, thresh_num, sigma_num] = num_labels
                if num_labels == 0:
                    median_area[t, thresh_num, sigma_num] = 0
                    continue
                # get median area using bincounts
                areas = xp.bincount(labeled_im.ravel())[1:]
                median_area[t, thresh_num, sigma_num] = xp.median(areas)

    std_area = xp.std(median_area, axis=0)
    std_num_components = xp.std(num_components, axis=0)
    mean_num_components = xp.mean(num_components, axis=0)
    # zscore normalize
    std_area = (std_area - xp.mean(std_area)) / xp.std(std_area)
    std_num_components = (std_num_components - xp.mean(std_num_components)) / xp.std(std_num_components)
    mean_num_components = (mean_num_components - xp.mean(mean_num_components)) / xp.std(mean_num_components)

    cost_matrix = std_area + std_num_components - 0.5 * mean_num_components
    # median_filter
    cost_matrix = ndi.median_filter(cost_matrix, size=3)
    cost_matrix[cost_matrix == 0] = 1e4

    min_indices = xp.unravel_index(xp.argmin(cost_matrix), cost_matrix.shape)

    # get corresponding sigma and threshold values
    best_thresh = int(thresh_matrix[int(min_indices[0])])
    best_sigma = float(sigma_matrix[int(min_indices[1])])

    final_seg = []
    for t in range(num_t):
        final_seg.append(ndi.gaussian_filter(all_im_bg_removed_timepoints[t], sigma=best_sigma) > best_thresh)
    final_seg = xp.asarray(final_seg)

    if device_type == 'cuda':
        final_seg = final_seg.get()
    tifffile.imwrite(os.path.join(output_dir, im_name), final_seg)

    return final_seg, output_dir


if __name__ == "__main__":
    # top_dir = r"C:\Users\austin\GitHub\nellie-simulations\separation\separation"
    # time_series_dir = r"C:\Users\austin\GitHub\nellie-simulations\separation\time_series"

    top_dir = time_series_dir = r"/Users/austin/GitHub/nellie-simulations/motion/linear"
    # time_series_dir = r"/Users/austin/GitHub/nellie-simulations/motion/angular"
    full_temporal = True
    run(top_dir, time_series_dir)
