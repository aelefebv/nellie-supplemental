import os
from multiprocessing import Pool
import tifffile
import ome_types
import pandas as pd
import numpy as np
import subprocess
import time

# implementing timeouts after 5 minutes or these simulation validations are going to take wayyyy too long on mitograph...
# plus, empirically, I've noticed long processing (> 1 min on these small images) means it did not segment properly.


def get_mitograph_output_path(im_dir, im_name):
    im = tifffile.imread(os.path.join(im_dir, im_name)).astype(np.uint16)
    im_name_no_ext = im_name.split('.')[0]
    all_output_path = os.path.join(im_dir, 'Mitograph', im_name_no_ext)
    return im, all_output_path


def convert(im, mitograph_input_dir):
    for i, frame in enumerate(im):
        new_dir = os.path.join(mitograph_input_dir, f"frame_{i}")
        os.makedirs(new_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(new_dir, f"frame_{i}.tif"), frame)
    return f"{mitograph_input_dir}/"


def run_mitograph(frame_dir, lateral_pixel_size, axial_pixel_size, mitograph_dir):
    # much of the Mitograph batch processing code has been adapted from MitoTNT's MitoGraph script:
    #  https://github.com/pylattice/MitoTNT/blob/f1fdc0dd465881af99205e928f4093d30c54e60c/helper_scripts/run_MitoGraph_parallel.ipynb
    command = (f"{os.path.join(mitograph_dir, 'MitoGraph')} "
               f"-xy {lateral_pixel_size} -z {axial_pixel_size} "
               f"-path {frame_dir}/")
    print(command)
    try:
        subprocess.run(command, shell=True, timeout=9999999, check=True)  # wait forever(ish)
    except subprocess.TimeoutExpired:
        print("Command timed out!")
        return "timeout"
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return "failure"
    # os.system(command)
    # reconstruct_image(frame_dir, mask=True)


def reconstruct_image(frame_dir, mask=False):
    mitograph_config_path = os.path.join(frame_dir, 'mitograph.config')
    with open(mitograph_config_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '-xy' in line:
                lateral_pixel_size = float(line.split('-xy')[-1].split('um')[0].strip())
            if '-z' in line:
                axial_pixel_size = float(line.split('-z')[-1].split('um')[0].strip())

    # find the coo file in output_path_dir:
    all_files = os.listdir(frame_dir)
    txt_file = [file for file in all_files if file.endswith('.txt')][0]
    tif_file = [file for file in all_files if file.endswith('.tif')][0]
    original_im = tifffile.imread(os.path.join(frame_dir, tif_file))

    bulk_nodes = pd.read_csv(os.path.join(frame_dir, txt_file), delimiter='\t')
    bulk_nodes['x_px'] = (bulk_nodes['x'] / lateral_pixel_size).astype(float)
    bulk_nodes['y_px'] = (bulk_nodes['y'] / lateral_pixel_size).astype(float)
    bulk_nodes['z_px'] = (bulk_nodes['z'] / axial_pixel_size).astype(float)
    bulk_nodes['width_px'] = (bulk_nodes['width_(um)'] / lateral_pixel_size).astype(float)

    # recreate image from bulk_nodes
    reconstructed_im = np.zeros_like(original_im)
    for i, row in bulk_nodes.iterrows():
        x_min = max(0, int(np.round(row['x_px'] - row['width_px'])))
        x_max = min(reconstructed_im.shape[2], int(np.round(row['x_px'] + row['width_px'])) + 1)
        x_len = x_max - x_min
        x_lims = (x_min, x_max)

        y_min = max(0, int(np.round(row['y_px'] - row['width_px'])))
        y_max = min(reconstructed_im.shape[1], int(np.round(row['y_px'] + row['width_px'])) + 1)
        y_len = y_max - y_min
        y_lims = (y_min, y_max)

        z_min = max(0, int(np.round(row['z_px'] - row['width_px'])))
        z_max = min(reconstructed_im.shape[0], int(np.round(row['z_px'] + row['width_px'])) + 1)
        z_len = z_max - z_min
        z_lims = (z_min, z_max)

        sphere = np.zeros((z_len, y_len, x_len), dtype=np.uint16)
        for z in range(z_len):
            for y in range(y_len):
                for x in range(x_len):
                    if (z - row['width_px']) ** 2 + (y - row['width_px']) ** 2 + (x - row['width_px']) ** 2 <= row[
                        'width_px'] ** 2:
                        if mask:
                            sphere[z, y, x] = 1
                        else:
                            sphere[z, y, x] = row['pixel_intensity']

        if mask:
            reconstructed_im[z_lims[0]:z_lims[1], y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]] += sphere
        else:
            current_vals = reconstructed_im[z_lims[0]:z_lims[1], y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]]
            current_vals[current_vals == 0] = sphere[current_vals == 0]
            current_vals[current_vals != 0] = np.mean([current_vals[current_vals != 0], sphere[current_vals != 0]])
            reconstructed_im[z_lims[0]:z_lims[1], y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]] = current_vals
    if mask:
        reconstructed_im = reconstructed_im > 0
    reconstructed_im = reconstructed_im[:, ::-1, :]  # flip y axis
    tifffile.imwrite(os.path.join(frame_dir, 'reconstructed.tif'), reconstructed_im)


def process_file(full_path):
    mitograph_dir = "/Users/austin/Desktop/MitoGraph"
    im_dir = os.path.dirname(full_path)
    im_name = os.path.basename(full_path)
    im_dir += "/"

    im, mitograph_input_path = get_mitograph_output_path(im_dir, im_name)

    if os.path.exists(mitograph_input_path):
        # print(f'{mitograph_input_path} already exists, skipping')
        return

    output_path = convert(im, mitograph_input_path)

    # Get metadata
    ome_xml = tifffile.tiffcomment(full_path)
    ome = ome_types.from_xml(ome_xml)
    lateral_px_size = ome.images[0].pixels.physical_size_x
    axial_px_size = ome.images[0].pixels.physical_size_z

    all_frames = sorted(os.listdir(output_path))
    all_frames = [os.path.join(output_path, frame) for frame in all_frames if os.path.isdir(os.path.join(output_path, frame))]
    start = time.time()
    for frame in all_frames:
        try:
            run_mitograph(frame, lateral_px_size, axial_px_size, mitograph_dir)
        except TimeoutError:
            print(f"Failed on {full_path}")
    print(f"Processing {full_path} took {time.time() - start} seconds")


def parallel_process_files(top_dirs):
    num_processes = os.cpu_count()

    all_files = []
    for top_dir in top_dirs:
        dir_files = os.listdir(top_dir)
        dir_files = [os.path.join(top_dir, file) for file in dir_files if file.endswith('.tif')]
        all_files.extend(dir_files)

    with Pool(num_processes) as pool:
        results = [pool.apply_async(process_file, (file,)) for file in all_files]
        for result in results:
            try:
                result.get(timeout=999999999)  # wait forever(ish)
            except:
                print("Processing file timed out!")


if __name__ == "__main__":
    top_dirs = [
        '/Users/austin/test_files/time_stuff',
    ]
    parallel_process_files(top_dirs)
