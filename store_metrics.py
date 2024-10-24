import argparse
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm
from pathlib import Path
import os
from functools import partial
from torch import Tensor, einsum
from monai.metrics import HausdorffDistanceMetric
from skimage.transform import resize
from typing import Iterable, Set
import math
import warnings

warnings.filterwarnings("ignore")

# Class names for segmentation
CLASS_NAMES = ['esophagus', 'heart', 'trachea', 'aorta']


def class_to_one_hot(seg: Tensor, num_classes: int) -> Tensor:
    """Convert segmentation labels to one-hot encoding."""
    assert is_subset(seg, list(range(num_classes))), (unique_elements(seg), num_classes)
    
    b, *img_shape = seg.shape
    device = seg.device
    one_hot_encoded = torch.zeros((b, num_classes, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)
    return one_hot_encoded


def unique_elements(tensor: Tensor) -> Set:
    """Get unique elements from a tensor."""
    return set(torch.unique(tensor).cpu().numpy())


def is_subset(tensor: Tensor, subset: Iterable) -> bool:
    """Check if elements of tensor are a subset of the given set."""
    return unique_elements(tensor).issubset(subset)


def intersection(a: Tensor, b: Tensor) -> Tensor:
    """Compute intersection of two binary tensors."""
    assert a.shape == b.shape
    assert is_subset(a, [0, 1])
    assert is_subset(b, [0, 1])
    
    res = a & b
    assert is_subset(res, [0, 1])
    return res


def compute_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    """Compute the Dice coefficient."""
    assert label.shape == pred.shape

    intersection_size = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dice_coefficients = (2 * intersection_size + smooth) / (sum_sizes + smooth)
    return dice_coefficients


dice_coefficient = partial(compute_dice, "bk...->bk")
dice_batch = partial(compute_dice, "bk...->k")


def compute_and_save_metrics(pred_vols, gt_vols, dest):
    """Compute and save evaluation metrics."""
    if pred_vols is None or gt_vols is None:
        raise ValueError("Predictions and ground truth volumes cannot be None.")

    dice_3d, hd = torch.zeros((2, len(pred_vols), 5))
    for i in tqdm(range(len(pred_vols))):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        p, g = pred_vols[i].to(device), gt_vols[i].to(device)
        dice_3d[i, :], hd[i, :] = compute_metrics(p, g)

    # Save metrics as numpy arrays
    np.save(os.path.join(dest, 'metric_DSC.npy'), dice_3d.cpu().numpy())
    np.save(os.path.join(dest, 'metric_hd95.npy'), hd.cpu().numpy())

    # Calculate mean metrics
    mean_dice_3d = np.mean(dice_3d.cpu().numpy(), axis=0)
    mean_hd = np.mean(hd.cpu().numpy(), axis=0)

    # Remove background class for metrics
    mean_dice_3d, mean_hd = mean_dice_3d[1:], mean_hd[1:]

    # Create dictionaries for class metrics
    dice_3d_class = {name_class: val for name_class, val in zip(CLASS_NAMES, mean_dice_3d)}
    hd_class = {name_class: val for name_class, val in zip(CLASS_NAMES, mean_hd)}

    print(f'Dice Coefficients: {dice_3d_class}, HD95: {hd_class}')


def compute_metrics(pred: Tensor, gt: Tensor):
    """Compute both Dice and Hausdorff metrics."""
    dice_3d = dice_batch(pred, gt)
    hd = hd95_batch(gt[None, ...].permute(0, 2, 3, 4, 1), pred[None, ...].permute(0, 2, 3, 4, 1), include_background=True)
    return dice_3d, hd


def resize_array(array, target_shape):
    """Resize array to target shape."""
    resized_array = resize(array, target_shape, mode="constant", preserve_range=True, anti_aliasing=False, order=0)
    return resized_array


def split_volumes_per_class(tensor, num_classes=5):
    """Split volumes into one-hot encoded format per class."""
    H, W, Z = tensor.shape
    resized_tensor = resize_array(tensor.cpu().numpy(), (H // 2, W // 2, Z))
    tensor = torch.from_numpy(resized_tensor).permute(2, 0, 1)
    return class_to_one_hot(tensor, num_classes)


def load_data(args, debug=True):
    """Load prediction and ground truth volumes from specified directories."""
    paths = [x for x in os.listdir(args.src) if x.endswith('.nii.gz')]
    patient_ids = [x.split('_')[1][:2] for x in paths] if not debug else ['01']
    print('Found patient IDs:', patient_ids)

    # Load nibabel files
    pred_nibs = [nib.load(os.path.join(args.src, f'Patient_{x}.nii.gz')) for x in patient_ids]
    gt_nibs = [nib.load(os.path.join(args.gt_src, f'Patient_{x}.nii.gz')) for x in patient_ids]

    # Load volumes as tensors
    pred_vols = [(torch.from_numpy(np.asarray(nib_file.dataobj))).type(torch.int64) for nib_file in pred_nibs]
    gt_vols = [(torch.from_numpy(np.asarray(nib_file.dataobj))).type(torch.int64) for nib_file in gt_nibs]

    # Validate loaded volumes
    if len(pred_vols) == 0 or len(gt_vols) == 0:
        raise ValueError("No prediction or ground truth volumes loaded.")

    # Split volumes per class
    print('Splitting volumes per class...')
    split_pred_vols = [split_volumes_per_class(vol) for vol in tqdm(pred_vols)]
    split_gt_vols = [split_volumes_per_class(vol) for vol in tqdm(gt_vols)]

    return split_pred_vols, split_gt_vols, pred_nibs, patient_ids


def hd95_batch(label: Tensor, pred: Tensor, include_background=False) -> Tensor:
    """
    Compute the 95th percentile Hausdorff Distance for a batch.
    
    Args:
        label: (batch, Classes, H, W, D) - one-hot encoded ground truth.
        pred: (batch, Classes, H, W, D) - one-hot encoded predictions.
        include_background: Whether to include the background class in calculations.
        
    Returns:
        Tensor: 95th percentile Hausdorff Distance for each class.
    """
    diagonal = math.sqrt(label.shape[2]**2 + label.shape[3]**2)
    hausdorff_metric = HausdorffDistanceMetric(include_background=include_background, percentile=95)
    
    score = hausdorff_metric(pred, label)
    score = torch.where(torch.isnan(score), diagonal, score)  # Replace NaN with diagonal distance
    return torch.mean(score, dim=0)  # Mean over batch


def main(args):
    """Main entry point for processing and computing metrics."""
    pred_volumes, gt_volumes, _, _ = load_data(args, debug=False)
    compute_and_save_metrics(pred_volumes, gt_volumes, args.dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate segmentation performance.')
    parser.add_argument('--src', type=Path, required=True, help='Path to folder containing volumes to process.')
    parser.add_argument('--dest', type=Path, required=True, help='Path to folder to save processed volumes.')
    parser.add_argument('--gt_src', type=Path, required=True, help='Path to folder containing ground truth volumes.')
    
    args = parser.parse_args()
    main(args)