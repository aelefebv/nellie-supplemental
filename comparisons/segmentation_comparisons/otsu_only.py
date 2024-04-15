import numpy as np
import tifffile

try:
    import cupy as xp
    device_type = 'cuda'
except ImportError:
    xp = np
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


if __name__ == "__main__":
    import os

    # top_dir = r"C:\Users\austin\GitHub\nellie-simulations\separation\separation"
    # top_dir = r"C:\Users\austin\GitHub\nellie-simulations\multi_grid\multigrid"
    top_dir = r"C:\Users\austin\GitHub\nellie-simulations\px_sizes\px_sizes"
    output_dir = os.path.join(top_dir, 'otsu')
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
        threshold, _ = otsu_threshold(im)
        final_seg = im > threshold
        if device_type == 'cuda':
            final_seg = final_seg.get()
        tifffile.imwrite(os.path.join(output_dir, file_names[file_num]), final_seg)
