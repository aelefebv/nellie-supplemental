import numpy as np
import tifffile

try:
    import cupy as xp
    device_type = 'cuda'
except ImportError:
    xp = np
    device_type = 'cpu'


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
    import os

    # top_dir = r"C:\Users\austin\GitHub\nellie-simulations\separation\separation"
    # top_dir = r"C:\Users\austin\GitHub\nellie-simulations\multi_grid\multigrid"
    top_dir = r"C:\Users\austin\GitHub\nellie-simulations\px_sizes\px_sizes"
    output_dir = os.path.join(top_dir, 'triangle')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_files = os.listdir(top_dir)
    file_names = [file for file in all_files if file.endswith('.tif')]
    for file_num, tif_file in enumerate(file_names):
        print(f'Processing file {file_num + 1} of {len(file_names)}')
        output_name = os.path.join(output_dir, file_names[file_num])
        if os.path.exists(output_name):
            print(f'already exists, skipping')
            continue
        filepath = os.path.join(top_dir, file_names[file_num])
        im = xp.asarray(tifffile.imread(filepath), dtype='float64')
        threshold = triangle_threshold(im)
        final_seg = im > threshold
        if device_type == 'cuda':
            final_seg = final_seg.get()
        tifffile.imwrite(os.path.join(output_dir, file_names[file_num]), final_seg)
