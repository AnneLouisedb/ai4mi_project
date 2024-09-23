import argparse
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from pathlib import Path
from typing import Dict
import re
import torch


def extract_files(
    base_folder: str, file_pattern: str, id_pattern: str
) -> Dict[str, sitk.Image]:
    base_path: Path = Path(base_folder).resolve()
    files: list[Path] = list(base_path.glob(file_pattern))
    id_regex: re.Pattern = re.compile(id_pattern)

    results: Dict[str, sitk.Image] = {}
    for file in files:
        match: re.Match | None = id_regex.search(str(file))
        if match:
            patient_id: str = match.group(1)
            results[patient_id] = sitk.ReadImage(str(file))
    return results


def batch_hausdorff_distance(pred: sitk.Image, target: sitk.Image) -> np.ndarray:
    """
    Calculate Hausdorff distance for predictions and targets using SimpleITK.
    
    Args:
    pred (sitk.Image): Predicted segmentation mask
    target (sitk.Image): Ground truth segmentation mask
    
    Returns:
    np.ndarray: Hausdorff distances and average Hausdorff distances for each class
    """
    # Convert SimpleITK images to numpy arrays
    pred_np = sitk.GetArrayFromImage(pred)
    pred_np = np.round(pred_np / 64).astype(int)
    target_np = sitk.GetArrayFromImage(target)
    
    # Assuming the images have 4 classes (background + 3 organs)
    num_classes = 4
    hausdorff_distances = np.zeros((num_classes, 2))
    
    organ_names = ['Esophagus', 'Heart', 'Trachea', 'Aorta'] # 'Background', ??
    
    for c in range(num_classes):
        # split on class
        pred_mask = (pred_np == c).astype(np.uint8)
        target_mask = (target_np == c).astype(np.uint8)
        
        if np.count_nonzero(pred_mask) > 0 and np.count_nonzero(target_mask) > 0:
            pred_sitk = sitk.GetImageFromArray(pred_mask)
            target_sitk = sitk.GetImageFromArray(target_mask)

            pred_sitk.CopyInformation(pred)
            target_sitk.CopyInformation(target)

            assert np.count_nonzero(sitk.GetArrayFromImage(target_sitk)) > 0, f"Ground truth file is empty for class {c}!"
            assert np.count_nonzero(sitk.GetArrayFromImage(pred_sitk)) > 0, f"Prediction file is empty for class {c}!"
            assert target_sitk.GetSpacing() == pred_sitk.GetSpacing(), f"Spacing of corresponding image files don't match for class {c}!"
            
            hausdorff_filter = sitk.HausdorffDistanceImageFilter()
            hausdorff_filter.Execute(pred_sitk, target_sitk)
            
            hausdorff_dist = hausdorff_filter.GetHausdorffDistance()
            avg_hausdorff_dist = hausdorff_filter.GetAverageHausdorffDistance()
            
            hausdorff_distances[c, 0] = hausdorff_dist
            hausdorff_distances[c, 1] = avg_hausdorff_dist
        else:
            hausdorff_distances[c, 0] = float('inf')
            hausdorff_distances[c, 1] = float('inf')
        
        print(f"{organ_names[c]}:")
        print(f"  Hausdorff Distance: {hausdorff_distances[c, 0]:.4f}")
        print(f"  Average Hausdorff Distance: {hausdorff_distances[c, 1]:.4f}")
    
    return hausdorff_distances


def plot_results(results, args):
    """
    Plot Hausdorff distances and average Hausdorff distances per organ and per image.
    
    Args:
    results (np.array): Hausdorff distances with shape (num_images, num_organs, 2)
                        where [:, :, 0] is Hausdorff and [:, :, 1] is average Hausdorff
    args: Arguments containing necessary information (e.g., organ names, output directory)
    """
    num_images, num_organs, _ = results.shape
    
    # Set up organ names (modify as per your dataset)
   
    # Set up organ names
    organ_names = ['Esophagus', 'Heart', 'Trachea', 'Aorta']
    assert num_organs == len(organ_names), f"Number of organs in results ({num_organs}) doesn't match the number of organ names ({len(organ_names)})"
    
    
    # Plot Hausdorff distances and average Hausdorff distances for each organ
    fig, axes = plt.subplots(num_organs, 2, figsize=(20, 6*num_organs))
    
    for i in range(num_organs):
        # Plot Hausdorff Distance
        axes[i, 0].boxplot(results[:, i, 0])
        axes[i, 0].set_title(f"Hausdorff Distance for {organ_names[i]}")
        axes[i, 0].set_ylabel("Distance")
        axes[i, 0].set_yscale('log')  # Log scale for better visualization
        
        # Plot Average Hausdorff Distance
        axes[i, 1].boxplot(results[:, i, 1])
        axes[i, 1].set_title(f"Average Hausdorff Distance for {organ_names[i]}")
        axes[i, 1].set_ylabel("Distance")
        axes[i, 1].set_yscale('log')  # Log scale for better visualization

    plt.tight_layout()

    save_path = Path('results/segthor') / Path(args.model) / "HD_per_organ.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Plot mean distances across all images for each organ
    mean_hausdorff = np.mean(results[:, :, 0], axis=0)
    mean_avg_hausdorff = np.mean(results[:, :, 1], axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.bar(organ_names, mean_hausdorff)
    ax1.set_title("Mean Hausdorff Distance per Organ")
    ax1.set_xlabel("Organ")
    ax1.set_ylabel("Mean Hausdorff Distance")
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(organ_names, mean_avg_hausdorff)
    ax2.set_title("Mean Average Hausdorff Distance per Organ")
    ax2.set_xlabel("Organ")
    ax2.set_ylabel("Mean Average Hausdorff Distance")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    save_path = Path('results/segthor') / Path(args.model) / "HD_mean_per_organ.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close('all')  # Close all figures to free up memory


 

def run(args):
    if args.dataset == "SEGTHOR":
        base_folder: str = "data/segthor_train/train"

    ground_truths: Dict[str, sitk.Image] = extract_files(
        base_folder, "Patient_*/GT.nii.gz", r"Patient_(\d+)"
    )

    # prediction files
    if args.model == "UNet":
        base_folder2: str = f"volumes/segthor/UNet/ce"

    predictions: Dict[str, sitk.Image] = extract_files(
        base_folder2, "Patient_*.nii.gz", r"Patient_(\d+)"
    )

  
    all_results = []

    for patient_id, pred_image in predictions.items():

        if patient_id in ground_truths:
            gt_image = ground_truths[patient_id]
            print(f"\nPatient ID: {patient_id}")
            print(type(pred_image))
            results = batch_hausdorff_distance(pred_image, gt_image)
            # print the hausdorff and average hausdoff per patient, per organ
            all_results.append(results)
            
    all_results = np.array(all_results)
    print(all_results)

    # Now you can call the plot_results function
    plot_results(all_results, args)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Hausdorff distance per image")
    parser.add_argument(
        "--source_scan_pattern",
        type=str,
        required=True,
        help='Pattern for ground truth scans, e.g., "data/segthor_train/train/{id_}/GT.nii.gz"',
    )
    parser.add_argument(
        "--prediction_folder",
        type=str,
        required=True,
        help="Path to the folder containing prediction files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="SEGTHOR",
        help="Path to the SEGTHOR dataset folder. Default is 'data/segthor_train/train'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="UNet",
        help="Path to the UNet model predictions folder. Default is 'volumes/segthor/UNet'.",
    )

    # python script.py --dataset data/segthor_train --model volumes/segthor/UNet

    return parser.parse_args()


if __name__ == "__main__":
    run(get_args())
