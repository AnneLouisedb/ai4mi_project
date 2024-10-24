import multiprocessing
import os
from copy import deepcopy
from multiprocessing import Pool
from typing import Tuple, List, Union, Optional

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, isfile
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json, determine_reader_writer_from_file_ending
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# the Evaluator class of the previous nnU-Net was great and all but man was it overengineered. Keep it simple
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion


def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)


def key_to_label_or_region(key: str):
    try:
        return int(key)
    except ValueError:
        key = key.replace('(', '')
        key = key.replace(')', '')
        split = key.split(',')
        return tuple([int(i) for i in split if len(i) > 0])


def save_summary_json(results: dict, output_file: str):
    """
    json does not support tuples as keys so we need to convert them ourselves
    """
    results_converted = deepcopy(results)
    # convert keys in mean metrics
    results_converted['mean'] = {label_or_region_to_key(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results_converted["metric_per_case"])):
        results_converted["metric_per_case"][i]['metrics'] = \
            {label_or_region_to_key(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=True)


def load_summary_json(filename: str):
    results = load_json(filename)
    # convert keys in mean metrics
    results['mean'] = {key_to_label_or_region(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results["metric_per_case"])):
        results["metric_per_case"][i]['metrics'] = \
            {key_to_label_or_region(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    return results


def labels_to_list_of_regions(labels: List[int]):
    return [(i,) for i in labels]


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


def compute_hausdorff(mask_ref: np.ndarray, mask_pred: np.ndarray, percentile: float = 100.0) -> float:
    """
    Computes the adjusted Hausdorff distance between predicted segmentation and ground truth for each class.
    Handles cases where segments are missing in either prediction or ground truth.
    Penalties are scaled based on the size of the missing or spurious segments.

    Parameters:
    - mask_ref: Ground truth binary mask (numpy array)
    - mask_pred: Predicted binary mask (numpy array)
    - percentile: Percentile of the distances to compute (default is 100 for maximum distance)

    Returns:
    - hausdorff_distance: The adjusted Hausdorff distance (float)
    """
    # Ensure masks are boolean
    mask_ref = mask_ref.astype(bool)
    mask_pred = mask_pred.astype(bool)

    # Compute total volume (number of voxels in the mask)
    total_voxels = mask_ref.size  # Total number of voxels in the volume

    # Calculate the volumes (number of voxels) of the masks
    pred_volume = np.sum(mask_pred)
    gt_volume = np.sum(mask_ref)

    # Compute scaling factors
    gt_scaling_factor = gt_volume / total_voxels
    pred_scaling_factor = pred_volume / total_voxels

    # Compute maximum possible distance (diagonal of the volume)
    max_coords = np.array(mask_ref.shape) - 1  # Subtract 1 because indices start at 0
    max_distance = np.linalg.norm(max_coords)

    # Extract surface voxels (edges) of the masks
    # Create a structuring element matching the mask dimensions
    struct = np.ones([3] * mask_ref.ndim, dtype=bool)

    edge_ref = mask_ref ^ binary_erosion(mask_ref, structure=struct)
    edge_pred = mask_pred ^ binary_erosion(mask_pred, structure=struct)

    # Get coordinates of surface voxels
    coords_ref = np.argwhere(edge_ref)
    coords_pred = np.argwhere(edge_pred)

    # Handle cases where one or both masks are empty
    if coords_ref.size == 0 and coords_pred.size == 0:
        # Both boundaries are empty; perfect agreement
        hausdorff_distance = 0.0
    elif coords_ref.size == 0 and coords_pred.size > 0:
        # GT segment is missing, prediction is present (False Positive)
        # Penalty proportional to predicted segment size
        hausdorff_distance = max_distance * pred_scaling_factor
    elif coords_ref.size > 0 and coords_pred.size == 0:
        # GT segment is present, prediction is missing (False Negative)
        # Penalty proportional to GT segment size
        hausdorff_distance = max_distance * gt_scaling_factor
    else:
        # Both segments are present; compute the Hausdorff distance
        # Build kd-trees for efficient nearest neighbor search
        kd_tree_ref = cKDTree(coords_ref)
        kd_tree_pred = cKDTree(coords_pred)

        # Compute nearest neighbor distances from reference to prediction
        distances_ref_to_pred, _ = kd_tree_pred.query(coords_ref, k=1)
        # Compute nearest neighbor distances from prediction to reference
        distances_pred_to_ref, _ = kd_tree_ref.query(coords_pred, k=1)

        # Compute the specified percentile of the distances
        hd_ref_to_pred = np.percentile(distances_ref_to_pred, percentile)
        hd_pred_to_ref = np.percentile(distances_pred_to_ref, percentile)

        # Hausdorff distance is the maximum of these two distances
        hausdorff_distance = max(hd_ref_to_pred, hd_pred_to_ref)

    return hausdorff_distance


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


def compute_metrics(reference_file: str, prediction_file: str, image_reader_writer: BaseReaderWriter,
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: int = None,
                    adjust_crf: bool = False) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    if adjust_crf:
        seg_ref = np.squeeze(seg_ref)
        seg_pred = np.squeeze(seg_pred)
        
        while seg_ref.ndim < seg_pred.ndim:
            seg_ref = np.expand_dims(seg_ref, axis=0)
        while seg_pred.ndim < seg_ref.ndim:
            seg_pred = np.expand_dims(seg_pred, axis=0)
        
        if seg_ref.shape != seg_pred.shape:
            raise ValueError(f"Cannot adjust shapes: seg_ref shape {seg_ref.shape}, seg_pred shape {seg_pred.shape}")

    if seg_ref.shape != seg_pred.shape:
        raise ValueError(f"Shape mismatch after adjustment: seg_ref shape {seg_ref.shape}, seg_pred shape {seg_pred.shape}")

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        
        # Compute Dice and IoU over the entire 3D volume
        dice = 2 * np.sum(mask_ref & mask_pred) / (np.sum(mask_ref) + np.sum(mask_pred))
        iou = np.sum(mask_ref & mask_pred) / np.sum(mask_ref | mask_pred)

        # Handle cases where both masks are empty
        if np.sum(mask_ref) == 0 and np.sum(mask_pred) == 0:
            dice = 1.0
            iou = 1.0

        results['metrics'][r]['Dice'] = dice
        results['metrics'][r]['IoU'] = iou

        # Hausdorff remains 3D
        results['metrics'][r]['Hausdorff'] = compute_hausdorff(mask_ref, mask_pred, percentile=100.0)
        results['metrics'][r]['Hausdorff95'] = compute_hausdorff(mask_ref, mask_pred, percentile=95.0)

        # Keep FN, FP, TN, TP, n_pred, n_ref as 2D
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp

    return results

def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              image_reader_writer: BaseReaderWriter,
                              file_ending: str,
                              regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                              ignore_label: int = None,
                              num_processes: int = default_num_processes,
                              chill: bool = True,
                              adjust_crf: bool = False) -> dict:
    """
    output_file must end with .json; can be None
    """
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_ref exist in folder_pred"
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        results = pool.starmap(
            compute_metrics,
            list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred),
                     [regions_or_labels] * len(files_pred),
                     [ignore_label] * len(files_pred),
                     [adjust_crf] * len(files_pred)))
        )

    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([i['metrics'][r][m] for i in results])

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.mean(values)

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    if output_file is not None:
        save_summary_json(result, output_file)
    return result


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False,
                               adjust_crf: bool = False):
    dataset_json = load_json(dataset_json_file)
    # get file ending
    file_ending = dataset_json['file_ending']

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                              num_processes, chill=chill, adjust_crf=adjust_crf)


def compute_metrics_on_folder_simple(folder_ref: str, folder_pred: str, labels: Union[Tuple[int, ...], List[int]],
                                     output_file: str = None,
                                     num_processes: int = default_num_processes,
                                     ignore_label: int = None,
                                     chill: bool = False):
    example_file = subfiles(folder_ref, join=True)[0]
    file_ending = os.path.splitext(example_file)[-1]
    rw = determine_reader_writer_from_file_ending(file_ending, example_file, allow_nonmatching_filename=True,
                                                  verbose=False)()
    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              labels, ignore_label=ignore_label, num_processes=num_processes, chill=chill)


def evaluate_folder_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('-pfile', type=str, required=True,
                        help='plans.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help="don't crash if folder_pred does not have all files that are present in folder_gt")
    parser.add_argument('--adjust_crf', action='store_true', help='Adjust CRF image dimensions during evaluation')
    args = parser.parse_args()
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np, chill=args.chill, adjust_crf=args.adjust_crf)


def evaluate_simple_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-l', type=int, nargs='+', required=True,
                        help='list of labels')
    parser.add_argument('-il', type=int, required=False, default=None,
                        help='ignore label')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help="don't crash if folder_pred does not have all files that are present in folder_gt")
    parser.add_argument('--adjust_crf', action='store_true', help='Adjust CRF image dimensions during evaluation')

    args = parser.parse_args()
    compute_metrics_on_folder_simple(args.gt_folder, args.pred_folder, args.l, args.o, args.np, args.il, chill=args.chill, adjust_crf=args.adjust_crf)


if __name__ == '__main__':
    evaluate_folder_entry_point()