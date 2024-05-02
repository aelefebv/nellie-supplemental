import numpy as np
from scipy.ndimage import gaussian_filter
from tifffile import tifffile
import ome_types


def add_noise(image, psf_sigma=0, gaussian_mean=0, gaussian_std=0.01, apply_poisson=True):
    noisy_image = image.copy().astype(np.float32)

    for frame_num, frame in enumerate(noisy_image):
        # PSF assume isotropic for comparisons purposes
        if psf_sigma > 0:
            frame = gaussian_filter(frame, sigma=psf_sigma)
        if gaussian_std > 0:
            normal_noise = np.random.normal(gaussian_mean, gaussian_std, frame.shape)
            frame += np.clip(normal_noise, 0, 65535)
        if apply_poisson:
            frame = np.clip(frame, 0, np.max(65535))
            frame = np.random.poisson(frame).astype(np.float32)

        frame = np.clip(frame, 0, 65535)
        # convert back to original dtype
        frame = frame.astype(image.dtype)
        noisy_image[frame_num] = frame

    return noisy_image


def save_ome_tif(path_im, data, dim_sizes):
    if len(data.shape) == 3:
        axes = 'ZYX'
    elif len(data.shape) == 4:
        axes = 'TZYX'
    else:
        axes = 'YX'
    tifffile.imwrite(
        path_im, data.astype(np.uint16), bigtiff=True, metadata={"axes": axes}, dtype=np.uint16
    )
    ome_xml = tifffile.tiffcomment(path_im)
    ome = ome_types.from_xml(ome_xml)
    ome.images[0].pixels.physical_size_x = dim_sizes['X']
    ome.images[0].pixels.physical_size_y = dim_sizes['Y']
    if 'Z' in dim_sizes:
        ome.images[0].pixels.physical_size_z = dim_sizes['Z']
    if 'T' in dim_sizes:
        ome.images[0].pixels.time_increment = dim_sizes['T']
    ome.images[0].pixels.type = "uint16"
    ome_xml = ome.to_xml()
    tifffile.tiffcomment(path_im, ome_xml)
