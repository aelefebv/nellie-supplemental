import os
import tifffile
from scipy import ndimage as ndi
import numpy as np
import pickle


def check_IoU(gt_label, output_im, is_label=False, min_area=None):
    gt_label = gt_label[5:-5, 5:-5, 5:-5]
    output_im = output_im[5:-5, 5:-5, 5:-5]
    num_sections = 10
    section_thickness = gt_label.shape[1] // num_sections
    # label with full 3d connectivity
    if is_label:
        output_label = output_im
    else:
        output_label, _ = ndi.label(output_im, structure=np.ones((3, 3, 3)))
    f1_array = np.zeros((num_sections, num_sections, num_sections), dtype=float)
    max_iou_array = np.zeros((num_sections, num_sections, num_sections), dtype=float)
    for thickness in range(num_sections):  # z
        for intensity in range(num_sections):  # y
            for length in range(num_sections):  # x
                label_partial = output_label[
                                thickness*section_thickness+1:(thickness+1)*section_thickness-1,
                                intensity*section_thickness+1:(intensity+1)*section_thickness-1,
                                length*section_thickness+1:(length+1)*section_thickness-1,
                                ]

                if min_area is not None:
                    flat_label_partial = label_partial.flatten()
                    bincount = np.bincount(flat_label_partial)
                    labels_over_min_area = np.where(bincount >= min_area)[0]
                    mask = np.isin(label_partial, labels_over_min_area)
                    label_partial = np.where(mask, label_partial, 0)

                gt_partial = gt_label[
                                thickness*section_thickness+1:(thickness+1)*section_thickness-1,
                                intensity*section_thickness+1:(intensity+1)*section_thickness-1,
                                length*section_thickness+1:(length+1)*section_thickness-1
                                ]
                gt_partial_coords = np.argwhere(gt_partial)

                # go through all label_partial labels and check IoU with gt_partial
                max_iou = 0
                tp_label = None
                unique_labels = np.unique(label_partial)
                if len(unique_labels) < 100:  # skip this, it's bad
                    for label in unique_labels:
                        if label == 0:
                            continue
                        label_coords = np.argwhere(label_partial == label)
                        # if none of the gt_partial_coords matches the label_coords, skip
                        if np.sum(np.isin(gt_partial_coords, label_coords).all(axis=1)) == 0:
                            continue

                        # test_label = label_partial == label
                        intersection = np.logical_and(gt_partial, label_partial == label)
                        union = np.logical_or(gt_partial, label_partial == label)
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

                f1_array[thickness, intensity, length] = f1
                max_iou_array[thickness, intensity, length] = max_iou

    return f1_array, max_iou_array


if __name__ == '__main__':
    raw_path = r"C:\Users\austin\GitHub\nellie-simulations\multi_grid\outputs"
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

    noiseless_im = tifffile.imread(os.path.join(raw_path, 'noiseless', 'no_noise.ome.tif'))
    noiseless_label, _ = ndi.label(noiseless_im)

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

    for num, tif in enumerate(raw_files):
        basename = os.path.basename(tif)[:os.path.basename(tif).find('.ome.tif')]
        print(f'Processing file {num + 1} of {len(raw_files)}')
        print(f'Looking for output for {basename}')
        for nellie_output in nellie_output_files:
            if basename in nellie_output:
                print(f'Found nellie output for {basename}')
                nellie_im = tifffile.imread(nellie_output)[0]
                f1, iou = check_IoU(noiseless_label, nellie_im, is_label=True)
                nellie_f1s[basename] = f1
                nellie_ious[basename] = iou
                break
        for mitograph_output in mitograph_output_files:
            if basename in mitograph_output:
                print(f'Found mitograph output for {basename}')
                mitograph_im_path = os.path.join(mitograph_output, 'frame_0', 'reconstructed.tif')
                mitograph_im = tifffile.imread(mitograph_im_path)
                f1, iou = check_IoU(noiseless_label, mitograph_im)
                mitograph_f1s[basename] = f1
                mitograph_ious[basename] = iou
                break
        for mitometer_output in mitometer_output_files:
            if basename in mitometer_output:
                print(f'Found mitometer output for {basename}')
                mitometer_im = tifffile.imread(mitometer_output)[0]
                f1, iou = check_IoU(noiseless_label, mitometer_im, min_area=4)
                mitometer_f1s[basename] = f1
                mitometer_ious[basename] = iou
                break
        for otsu_output in otsu_output_files:
            if basename in otsu_output:
                print(f'Found otsu output for {basename}')
                otsu_im = tifffile.imread(otsu_output)[0]
                f1, iou = check_IoU(noiseless_label, otsu_im)
                otsu_f1s[basename] = f1
                otsu_ious[basename] = iou
                break
        for triangle_output in triangle_output_files:
            if basename in triangle_output:
                print(f'Found triangle output for {basename}')
                triangle_im = tifffile.imread(triangle_output)[0]
                f1, iou = check_IoU(noiseless_label, triangle_im)
                triangle_f1s[basename] = f1
                triangle_ious[basename] = iou
                break

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
