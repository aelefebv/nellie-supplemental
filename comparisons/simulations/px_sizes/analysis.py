import os
import tifffile
from scipy import ndimage as ndi
import numpy as np
import pickle


def check_IoU(gt_label, output_im, is_label=False, min_area=None):
    try:
        # label with full 3d connectivity
        if is_label:
            output_label = output_im
        else:
            output_label, _ = ndi.label(output_im, structure=np.ones((3, 3, 3)))

        if min_area is not None:
            flat_label_partial = output_label.flatten()
            bincount = np.bincount(flat_label_partial)
            labels_over_min_area = np.where(bincount >= min_area)[0]
            mask = np.isin(output_label, labels_over_min_area)
            output_label = np.where(mask, output_label, 0)

        gt_coords = np.argwhere(gt_label)

        # go through all label_partial labels and check IoU with gt_partial
        max_iou = 0
        tp_label = None
        unique_labels = np.unique(output_label)
        if len(unique_labels) < 100:  # skip this, it's bad
            for label in unique_labels:
                if label == 0:
                    continue
                label_coords = np.argwhere(output_label == label)
                # if none of the gt_partial_coords matches the label_coords, skip
                if np.sum(np.isin(gt_coords, label_coords).all(axis=1)) == 0:
                    continue

                intersection = np.logical_and(gt_label, output_label == label)
                union = np.logical_or(gt_label, output_label == label)
                iou = np.sum(intersection) / np.sum(union)
                if iou > max_iou:
                    max_iou = iou
                    tp_label = label

        fp = len(unique_labels[unique_labels != tp_label])-1  # -1 to account for 0 label
        fn = 0
        tp = 1
        if tp_label is None:
            fn = 1
            tp = 0

        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        if max_iou < 0.1:  # Indicating no good match
            f1 = 0
    except Exception as e:
        print(f'Error: {e}')
        f1 = 0
        max_iou = 0

    return f1, max_iou


if __name__ == '__main__':
    raw_path = r"C:\Users\austin\GitHub\nellie-simulations\px_sizes\outputs"
    nellie_outputs = os.path.join(raw_path, 'nellie_output')
    mitograph_outputs = os.path.join(raw_path, 'mitograph')
    mitometer_outputs = os.path.join(raw_path, 'mitometer')
    otsu_outputs = os.path.join(raw_path, 'otsu')
    triangle_outputs = os.path.join(raw_path, 'triangle')

    raw_files = os.listdir(raw_path)
    raw_files = [os.path.join(raw_path, file) for file in raw_files if file.endswith('.tif')]

    nellie_output_files = os.listdir(nellie_outputs)
    nellie_output_files = [os.path.join(nellie_outputs, file) for file in nellie_output_files if file.endswith('skel_relabelled.ome.tif')]

    mitograph_output_files = os.listdir(mitograph_outputs)
    mitograph_output_files = [os.path.join(mitograph_outputs, file) for file in mitograph_output_files if os.path.isdir(os.path.join(mitograph_outputs, file))]

    mitometer_output_files = os.listdir(mitometer_outputs)
    mitometer_output_files = [os.path.join(mitometer_outputs, file) for file in mitometer_output_files if file.endswith('.tif')]

    otsu_output_files = os.listdir(otsu_outputs)
    otsu_output_files = [os.path.join(otsu_outputs, file) for file in otsu_output_files if file.endswith('.tif')]

    triangle_output_files = os.listdir(triangle_outputs)
    triangle_output_files = [os.path.join(triangle_outputs, file) for file in triangle_output_files if file.endswith('.tif')]

    noiseless_files = os.listdir(os.path.join(raw_path, 'noiseless'))
    noiseless_files = [os.path.join(raw_path, 'noiseless', file) for file in noiseless_files if file.endswith('.tif')]

    nellie_f1s = {}
    nellie_ious = {}
    mitograph_f1s = {}
    mitograph_ious = {}
    mitometer_f1s = {}
    mitometer_ious = {}
    otsu_f1s = {}
    otsu_ious = {}
    triangle_f1s = {}
    triangle_ious = {}

    total_run_num = 0
    for im_path in noiseless_files:
        name_start = os.path.basename(im_path).find('no_noise-')+len('no_noise-')
        basename = os.path.basename(im_path)[name_start:os.path.basename(im_path).find('.ome.tif')]
        noiseless_label, _ = ndi.label(tifffile.imread(im_path))
        print(f'Processing file {total_run_num + 1} of {len(noiseless_files)}')
        # get file in nellie_output_files that contains name
        for nellie_output in nellie_output_files:
            if basename in nellie_output:
                name = f"{os.path.basename(nellie_output).split('-')[0]}-{basename}"
                print(f'Found nellie output for {name}')
                nellie_im = tifffile.imread(nellie_output)[0]
                f1, iou = check_IoU(noiseless_label, nellie_im, is_label=True)
                nellie_f1s[name] = f1
                nellie_ious[name] = iou
        for mitograph_output in mitograph_output_files:
            if basename in mitograph_output:
                name = f"{os.path.basename(mitograph_output).split('-')[0]}-{basename}"
                print(f'Found mitograph output for {name}')
                mitograph_im_path = os.path.join(mitograph_output, 'frame_0', 'reconstructed.tif')
                if not os.path.exists(mitograph_im_path):
                    continue
                mitograph_im = tifffile.imread(mitograph_im_path)
                f1, iou = check_IoU(noiseless_label, mitograph_im)
                mitograph_f1s[name] = f1
                mitograph_ious[name] = iou
        for mitometer_output in mitometer_output_files:
            if basename in mitometer_output:
                name = f"{os.path.basename(mitometer_output).split('-')[0]}-{basename}"
                print(f'Found mitometer output for {name}')
                mitometer_im = tifffile.imread(mitometer_output)[0]
                f1, iou = check_IoU(noiseless_label, mitometer_im, min_area=4)
                mitometer_f1s[name] = f1
                mitometer_ious[name] = iou
        for otsu_output in otsu_output_files:
            if basename in otsu_output:
                name = f"{os.path.basename(otsu_output).split('-')[0]}-{basename}"
                print(f'Found otsu output for {name}')
                otsu_im = tifffile.imread(otsu_output)[0]
                f1, iou = check_IoU(noiseless_label, otsu_im)
                otsu_f1s[name] = f1
                otsu_ious[name] = iou
        for triangle_output in triangle_output_files:
            if basename in triangle_output:
                name = f"{os.path.basename(triangle_output).split('-')[0]}-{basename}"
                print(f'Found triangle output for {name}')
                triangle_im = tifffile.imread(triangle_output)[0]
                f1, iou = check_IoU(noiseless_label, triangle_im)
                triangle_f1s[name] = f1
                triangle_ious[name] = iou
        total_run_num += 1

        nellie_stats = {
            'f1': nellie_f1s,
            'iou': nellie_ious
        }
        with open(os.path.join(nellie_outputs, 'nellie_stats.pkl'), 'wb') as f:
            pickle.dump(nellie_stats, f)

        mitograph_stats = {
            'f1': mitograph_f1s,
            'iou': mitograph_ious
        }
        with open(os.path.join(mitograph_outputs, 'mitograph_stats.pkl'), 'wb') as f:
            pickle.dump(mitograph_stats, f)

        mitometer_stats = {
            'f1': mitometer_f1s,
            'iou': mitometer_ious
        }
        with open(os.path.join(mitometer_outputs, 'mitometer_stats.pkl'), 'wb') as f:
            pickle.dump(mitometer_stats, f)

        otsu_stats = {
            'f1': otsu_f1s,
            'iou': otsu_ious
        }
        with open(os.path.join(otsu_outputs, 'otsu_stats.pkl'), 'wb') as f:
            pickle.dump(otsu_stats, f)

        triangle_stats = {
            'f1': triangle_f1s,
            'iou': triangle_ious
        }
        with open(os.path.join(triangle_outputs, 'triangle_stats.pkl'), 'wb') as f:
            pickle.dump(triangle_stats, f)
