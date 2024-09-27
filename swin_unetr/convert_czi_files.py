import czifile
import napari
import tifffile
import os
top_top_dir = r"D:\test_files\aics_dataset\raw"
all_topdirs = [os.path.join(top_top_dir, f) for f in os.listdir(top_top_dir) if os.path.isdir(os.path.join(top_top_dir, f))]
# remove the 'done' directory
all_topdirs = [d for d in all_topdirs if 'done' not in d]

for topdir in all_topdirs:
    topdir_basename = os.path.basename(topdir)
    structure_name = topdir_basename.split('_')[-1]
    print(f"Processing structure: {structure_name}")
    output_dir = rf"D:\test_files\aics_dataset\{structure_name}"
    ch = 3
    test = False

    os.makedirs(output_dir, exist_ok=True)

    viewer = napari.Viewer() if test else None
    all_czi_files = [os.path.join(topdir, f) for f in os.listdir(topdir) if f.endswith('.czi')]

    if test:
        all_czi_files = [all_czi_files[0]]

    for im_num, path in enumerate(all_czi_files):
        print(f"Processing image {im_num + 1} of {len(all_czi_files)}")
        im_name_no_ext = os.path.basename(path).split('.')[0]
        czi = czifile.CziFile(path)
        data = czi.asarray()

        # Remove dimensions with only one value
        data = data.squeeze()
        if test:
            viewer.add_image(data)
            continue

        good_im = data[ch]

        output_name = os.path.join(output_dir, f"{im_name_no_ext}_{structure_name}.ome.tif")

        # Save as OME-TIFF with metadata
        tifffile.imwrite(
            output_name,
            good_im,
            ome=True,
            metadata={
                'axes': 'ZYX',
                'PhysicalSizeX': 0.108,
                'PhysicalSizeY': 0.108,
                'PhysicalSizeZ': 0.29,
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeYUnit': 'µm',
                'PhysicalSizeZUnit': 'µm',
            }
        )

    if test:
        napari.run()
