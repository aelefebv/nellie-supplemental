from itertools import combinations_with_replacement

import numpy as np
import tifffile

try:
    import cupy as xp
    import cupyx.scipy.ndimage as ndi
    device_type = 'cuda'
except ImportError:
    xp = np
    import scipy.ndimage as ndi
    device_type = 'cpu'

# input TIFF z-stack
# output vesselness-stack


# paper states from sigma min to sigma min (presumably should be sigma max)
# the code has default 1 to 1.5 with 6 steps
sigma_min = 1.0
sigma_max = 1.5
Nsigma = 6
sigmas = np.linspace(sigma_min, sigma_max, Nsigma)

# Frangi parameters are hardcoded as follows (from supplemental)
a = 0.5
b = 0.5
c = 500

a_sq = a ** 2
b_sq = b ** 2
c_sq = c ** 2


def run_frame(stack):
    # convert 16-bit to 8-bit image as done in paper
    if stack.dtype == 'uint16':
        stack = np.round(stack / 256).astype('uint8')

    print(f'Running frangi filter.')
    vesselness = xp.zeros_like(stack, dtype='float64')
    temp = xp.zeros_like(stack, dtype='float64')
    for sigma_num, sigma in enumerate(sigmas):
        gauss_volume = gauss_filter(sigma, stack)

        h_mask, hessian_matrices = compute_hessian(gauss_volume)
        eigenvalues = compute_chunkwise_eigenvalues(hessian_matrices.astype('float'))

        temp[h_mask] = filter_hessian(eigenvalues)  # only set values where h_mask is true (from frobenius norm)

        max_indices = temp > vesselness
        vesselness[max_indices] = temp[max_indices]

    return vesselness


# Filter with no anisotropic consideration, which is fine since comparison data is all isotropic
def gauss_filter(sigma, stack):
    gauss_volume = xp.asarray(stack, dtype='double')
    print(f'Gaussian filtering with {sigma=}.')

    gauss_volume = ndi.gaussian_filter(gauss_volume, sigma=sigma,
                                       mode='reflect', cval=0.0, truncate=3).astype('double')
    return gauss_volume


def compute_hessian(image):
    gradients = xp.gradient(image)
    axes = range(image.ndim)
    h_elems = xp.array([xp.gradient(gradients[ax0], axis=ax1).astype('float16')
                        for ax0, ax1 in combinations_with_replacement(axes, 2)])
    frob_norm = xp.linalg.norm(h_elems, axis=0)
    frobenius_max = xp.sqrt(xp.max(frob_norm))  # defaults to frobenius norm
    h_mask = frob_norm >= frobenius_max

    # masking before eigenvalue computation. same effect, but faster than what they describe in the paper
    if device_type == 'cuda':
        hxx, hxy, hxz, hyy, hyz, hzz = [elem[..., np.newaxis, np.newaxis] for elem in h_elems[:, h_mask].get()]
    else:
        hxx, hxy, hxz, hyy, hyz, hzz = [elem[..., np.newaxis, np.newaxis] for elem in h_elems[:, h_mask]]
    hessian_matrices = np.concatenate([
        np.concatenate([hxx, hxy, hxz], axis=-1),
        np.concatenate([hxy, hyy, hyz], axis=-1),
        np.concatenate([hxz, hyz, hzz], axis=-1)
    ], axis=-2)

    return h_mask, hessian_matrices


def compute_chunkwise_eigenvalues(hessian_matrices, chunk_size=1E6):
    chunk_size = int(chunk_size)
    total_voxels = len(hessian_matrices)

    eigenvalues_list = []

    if chunk_size is None:  # chunk size is entire vector
        chunk_size = total_voxels

    # iterate over chunks
    for start_idx in range(0, total_voxels, int(chunk_size)):
        end_idx = min(start_idx + chunk_size, total_voxels)
        gpu_chunk = xp.array(hessian_matrices[start_idx:end_idx])
        chunk_eigenvalues = xp.linalg.eigvalsh(gpu_chunk)
        eigenvalues_list.append(chunk_eigenvalues)

    # concatenate all the eigval chunks and reshape to the original spatial structure
    eigenvalues_flat = xp.concatenate(eigenvalues_list, axis=0)
    sort_order = xp.argsort(xp.abs(eigenvalues_flat), axis=1)
    eigenvalues_flat = xp.take_along_axis(eigenvalues_flat, sort_order, axis=1)

    return eigenvalues_flat


def filter_hessian(eigenvalues):
    ra_sq = (xp.abs(eigenvalues[:, 1]) / xp.abs(eigenvalues[:, 2])) ** 2
    rb_sq = (xp.abs(eigenvalues[:, 1]) / xp.sqrt(xp.abs(eigenvalues[:, 1] * eigenvalues[:, 2]))) ** 2
    s_sq = (xp.sqrt((eigenvalues[:, 0] ** 2) + (eigenvalues[:, 1] ** 2) + (eigenvalues[:, 2] ** 2))) ** 2
    filtered_im = (1 - xp.exp(-(ra_sq / (2*a_sq)))) * (xp.exp(-(rb_sq / (2*b_sq)))) * \
                  (1 - xp.exp(-(s_sq / (2*c_sq))))
    filtered_im[eigenvalues[:, 2] > 0] = 0
    filtered_im[eigenvalues[:, 1] > 0] = 0
    filtered_im = xp.nan_to_num(filtered_im, False, 1)
    return filtered_im


def divergence(vesselness):
    grad_x, grad_y, grad_z = xp.gradient(vesselness)
    grad_mag = xp.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)

    grad_mag_over_0 = grad_mag > 0
    norm_grad_x = grad_x / grad_mag
    norm_grad_y = grad_y / grad_mag
    norm_grad_z = grad_z / grad_mag
    norm_grad_x[~grad_mag_over_0] = 0
    norm_grad_y[~grad_mag_over_0] = 0
    norm_grad_z[~grad_mag_over_0] = 0

    div = xp.gradient(norm_grad_x)[0] + xp.gradient(norm_grad_y)[1] + xp.gradient(norm_grad_z)[2]
    div *= -1
    # only keep values above or equal to 1/6 as per the paper, make it a mask
    div = div >= 1/6

    return div


if __name__ == "__main__":
    import os

    top_dir = r"C:\Users\austin\GitHub\nellie-simulations\separation\separation"
    output_dir = os.path.join(top_dir, 'mitograph')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_files = os.listdir(top_dir)
    file_names = [file for file in all_files if not os.path.isdir(os.path.join(top_dir, file))]
    all_files = [os.path.join(top_dir, file) for file in all_files if not os.path.isdir(os.path.join(top_dir, file))]
    for file_num, tif_file in enumerate(all_files):
        # for ch in range(1):
        print(f'Processing file {file_num + 1} of {len(all_files)}')
        im = xp.asarray(tifffile.imread(tif_file), dtype='float64')
        vesselness = run_frame(im)
        div = divergence(vesselness)
        if device_type == 'cuda':
            div = div.get()
        tifffile.imwrite(os.path.join(output_dir, file_names[file_num]), div)
