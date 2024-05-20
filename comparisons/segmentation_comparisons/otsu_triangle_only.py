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


def otsu_threshold(matrix, nbins=256):
    # gpu version of skimage.filters.threshold_otsu
    counts, bin_edges = xp.histogram(matrix.reshape(-1), bins=nbins, range=(matrix.min(), matrix.max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    counts = counts / xp.sum(counts)

    weight1 = xp.cumsum(counts)
    mean1 = xp.cumsum(counts * bin_centers) / weight1
    if device_type == 'mps':
        weight2 = xp.cumsum(xp.flip(counts, dims=[0]))
        weight2 = xp.flip(weight2, dims=[0])
        flipped_counts_bin_centers = xp.flip(counts * bin_centers, dims=[0])
        cumsum_flipped = xp.cumsum(flipped_counts_bin_centers)
        mean2 = xp.flip(cumsum_flipped / xp.flip(weight2, dims=[0]), dims=[0])
    else:
        weight2 = xp.cumsum(counts[::-1])[::-1]
        mean2 = (xp.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = xp.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold, variance12[idx]


def triangle_threshold(matrix, nbins=256):
    # gpu version of skimage.filters.threshold_triangle
    hist, bin_edges = xp.histogram(matrix.reshape(-1), bins=nbins, range=(xp.min(matrix), xp.max(matrix)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    hist = hist / xp.sum(hist)

    arg_peak_height = xp.argmax(hist)
    peak_height = hist[arg_peak_height]
    arg_low_level, arg_high_level = xp.flatnonzero(hist)[[0, -1]]

    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        if device_type == 'mps':
            hist = xp.flip(hist, dims=[0])
        else:
            hist = xp.flip(hist, axis=0)

        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1
    del (arg_high_level)

    width = arg_peak_height - arg_low_level
    x1 = xp.arange(width)
    y1 = hist[x1 + arg_low_level]

    norm = xp.sqrt(peak_height ** 2 + width ** 2)
    peak_height = peak_height / norm
    width = width / norm

    length = peak_height * x1 - width * y1

    arg_level = xp.argmax(length) + arg_low_level if length.size > 0 else 0

    if flip:
        arg_level = nbins - arg_level - 1

    if arg_level < 0:
        arg_level = 0

    return bin_centers[arg_level]


if __name__ == "__main__":
    top_dirs = [
        "/Users/austin/GitHub/nellie-simulations/multi_grid/outputs",
        "/Users/austin/GitHub/nellie-simulations/px_sizes/outputs",
        "/Users/austin/GitHub/nellie-simulations/separation/outputs",
    ]

    for top_dir in top_dirs:
        output_otsu_dir = os.path.join(top_dir, 'otsu')
        output_triangle_dir = os.path.join(top_dir, 'triangle')
        if not os.path.exists(output_otsu_dir):
            os.makedirs(output_otsu_dir)
        if not os.path.exists(output_triangle_dir):
            os.makedirs(output_triangle_dir)

        all_files = os.listdir(top_dir)
        file_names = [file for file in all_files if file.endswith('.tif')]
        for file_num, tif_file in enumerate(file_names):
            print(f'Processing file {file_num + 1} of {len(file_names)}')
            output_otsu_name = os.path.join(output_otsu_dir, file_names[file_num])
            output_triangle_name = os.path.join(output_triangle_dir, file_names[file_num])
            if os.path.exists(output_otsu_name):
                print(f'already exists, skipping')
                continue
            filepath = os.path.join(top_dir, file_names[file_num])
            im = xp.asarray(tifffile.imread(filepath), dtype='float64')
            # small gaussian filter
            im = ndi.gaussian_filter(im, sigma=0.5)
            otsu, _ = otsu_threshold(im)
            triangle = triangle_threshold(im)
            final_otsu = im > otsu
            final_triangle = im > triangle
            if device_type == 'cuda':
                final_otsu = final_otsu.get()
                final_triangle = final_triangle.get()
            tifffile.imwrite(os.path.join(output_otsu_dir, file_names[file_num]), final_otsu)
            tifffile.imwrite(os.path.join(output_triangle_dir, file_names[file_num]), final_triangle)

