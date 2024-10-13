import argparse
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Dict, List
import re

def extract_files(
    base_folder: str, file_pattern: str, id_pattern: str
) -> Dict[str, sitk.Image]:
    """
    Extract and read image files matching a specific pattern and extract patient IDs using regex.

    Args:
        base_folder (str): The base directory to search for files.
        file_pattern (str): The glob pattern to match files.
        id_pattern (str): The regex pattern to extract patient IDs.

    Returns:
        Dict[str, sitk.Image]: A dictionary mapping patient IDs to their corresponding images.
    """
    base_path: Path = Path(base_folder).resolve()
    files: list[Path] = list(base_path.glob(file_pattern))
    id_regex: re.Pattern = re.compile(id_pattern)

    results: Dict[str, sitk.Image] = {}
    for file in files:
        match: re.Match | None = id_regex.search(str(file))
        if match:
            patient_id: str = match.group(1)
            try:
                results[patient_id] = sitk.ReadImage(str(file))
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return results


def print_image_info(image: sitk.Image, image_name: str):
    """
    Print detailed information about a SimpleITK image.

    Args:
        image (sitk.Image): The image to inspect.
        image_name (str): A descriptive name for the image.
    """
    print(f"--- {image_name} ---")
    print(f"Size: {image.GetSize()}")
    print(f"Spacing: {image.GetSpacing()}")
    print(f"Origin: {image.GetOrigin()}")
    print(f"Direction: {image.GetDirection()}")
    print(f"Pixel Type: {image.GetPixelIDTypeAsString()}\n")


def resample_image(reference_image: sitk.Image, moving_image: sitk.Image, interpolator=sitk.sitkNearestNeighbor) -> sitk.Image:
    """
    Resample moving_image to the space of reference_image.

    Args:
        reference_image (sitk.Image): The image whose space will be used as reference.
        moving_image (sitk.Image): The image to be resampled.
        interpolator: The interpolation method. Default is sitkNearestNeighbor for label images.

    Returns:
        sitk.Image: The resampled moving image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampled_image = resampler.Execute(moving_image)
    return resampled_image


def batch_hausdorff_distance(pred: sitk.Image, target: sitk.Image) -> np.ndarray:
    """
    Calculate the Hausdorff distance and 95th percentile Hausdorff distance for predictions and targets using SimpleITK.

    Args:
        pred (sitk.Image): Predicted segmentation mask.
        target (sitk.Image): Ground truth segmentation mask.

    Returns:
        np.ndarray: Hausdorff distances and HD95 for each class.
    """
    # Convert SimpleITK images to numpy arrays
    pred_np = sitk.GetArrayFromImage(pred)
    target_np = sitk.GetArrayFromImage(target)

    # Normalize prediction labels by dividing by 64 to match ground truth labels
    pred_np = np.round(pred_np / 64).astype(int)

    # Verify normalization
    unique_pred = np.unique(pred_np)
    unique_target = np.unique(target_np)
    print(f"Unique labels in prediction after normalization: {unique_pred}")
    print(f"Unique labels in ground truth: {unique_target}")

    # Define classes (including background)
    num_classes = 5  # 0: Background, 1-4: Organs
    hausdorff_distances = np.full((num_classes, 2), np.nan)  # Initialize with NaN

    organ_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']
    colors = ['black', 'green', 'yellow', 'red', 'blue']

    for c in range(num_classes):
        pred_mask = (pred_np == c).astype(np.uint8)
        target_mask = (target_np == c).astype(np.uint8)

        if c == 0:
            # Optionally skip background if it's not of interest
            pass  # Keep background for now

        if np.count_nonzero(pred_mask) == 0 and np.count_nonzero(target_mask) == 0:
            print(f"{organ_names[c]}: Both prediction and ground truth are empty.")
            hausdorff_distances[c, :] = np.nan  # Assign NaN for empty classes
            continue
        elif np.count_nonzero(pred_mask) == 0 or np.count_nonzero(target_mask) == 0:
            print(f"{organ_names[c]}: One of the masks is empty.")
            hausdorff_distances[c, :] = np.nan
            continue

        pred_sitk = sitk.GetImageFromArray(pred_mask)
        target_sitk = sitk.GetImageFromArray(target_mask)

        pred_sitk.CopyInformation(pred)
        target_sitk.CopyInformation(target)

        # Compute Hausdorff Distance
        hausdorff_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_filter.Execute(target_sitk, pred_sitk)

        # Get the Hausdorff distance
        hausdorff_dist = hausdorff_filter.GetHausdorffDistance()
        hausdorff_distances[c, 0] = hausdorff_dist

        # Compute 95th percentile Hausdorff Distance (HD95)
        y_true_contour = sitk.LabelContour(target_sitk, False)
        y_pred_contour = sitk.LabelContour(pred_sitk, False)
        y_true_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(y_true_contour, squaredDistance=False, useImageSpacing=True))
        y_pred_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(y_pred_contour, squaredDistance=False, useImageSpacing=True))

        # Extract distances where contours are present
        dist_y_pred = sitk.GetArrayViewFromImage(y_pred_distance_map)[sitk.GetArrayViewFromImage(y_true_contour) > 0]
        dist_y_true = sitk.GetArrayViewFromImage(y_true_distance_map)[sitk.GetArrayViewFromImage(y_pred_contour) > 0]

        if dist_y_true.size == 0 or dist_y_pred.size == 0:
            hausdorff_distances[c, 1] = np.nan
            print(f"{organ_names[c]}: Insufficient data for HD95 calculation.")
        else:
            hd95 = (np.percentile(dist_y_true, 95) + np.percentile(dist_y_pred, 95)) / 2.0
            hausdorff_distances[c, 1] = hd95

        print(f"{organ_names[c]}:")
        print(f"  95th Percentile Hausdorff Distance: {hausdorff_distances[c, 1]:.4f}")
        print(f"  Hausdorff Distance: {hausdorff_distances[c, 0]:.4f}")

    return hausdorff_distances


def plot_hd_per_organ(patient_ids: List[str], all_results: np.ndarray, bf: str):
    """
    Plot a line graph of Hausdorff distances (HD) for each organ across patients, excluding background.

    Args:
        patient_ids (List[str]): List of patient identifiers.
        all_results (np.ndarray): Array of shape (num_patients, num_organs, 2) containing HD and HD95.
        bf (str): Base folder identifier for saving plots.
    """
    num_patients, num_organs, num_metrics = all_results.shape  # num_metrics=2 (HD, HD95)

    # Set up organ names
    organ_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']
    colors = ['black', 'green', 'yellow', 'red', 'blue']

    # Initialize plot
    plt.figure(figsize=(14, 8))

    # Plot HD for each organ excluding background
    for c in range(1, num_organs):  # Start from 1 to exclude background
        hd = all_results[:, c, 0]

        # Handle NaN values by masking
        hd = np.where(np.isnan(hd), np.nan, hd)

        # Plot HD
        plt.plot(patient_ids, hd, label=f"{organ_names[c]} HD", color=colors[c], linestyle='-')

    # Customize plot
    plt.xlabel('Patient ID')
    plt.ylabel('Hausdorff Distance (mm)')
    plt.title('Hausdorff Distance (HD) per Organ Across Patients')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Create the directory if it doesn't exist
    save_dir = Path('results/segthor') / Path(bf)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "HD_per_organ_across_patients.png"

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hausdorff Distance plot saved to {save_path}")


def plot_hd95_per_organ(patient_ids: List[str], all_results: np.ndarray, bf: str):
    """
    Plot a line graph of 95th Percentile Hausdorff distances (HD95) for each organ across patients, excluding background.

    Args:
        patient_ids (List[str]): List of patient identifiers.
        all_results (np.ndarray): Array of shape (num_patients, num_organs, 2) containing HD and HD95.
        bf (str): Base folder identifier for saving plots.
    """
    num_patients, num_organs, num_metrics = all_results.shape  # num_metrics=2 (HD, HD95)

    # Set up organ names
    organ_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']
    colors = ['black', 'green', 'yellow', 'red', 'blue']

    # Initialize plot
    plt.figure(figsize=(14, 8))

    # Plot HD95 for each organ excluding background
    for c in range(1, num_organs):  # Start from 1 to exclude background
        hd95 = all_results[:, c, 1]

        # Handle NaN values by masking
        hd95 = np.where(np.isnan(hd95), np.nan, hd95)

        # Plot HD95
        plt.plot(patient_ids, hd95, label=f"{organ_names[c]} HD95", color=colors[c], linestyle='--')

    # Customize plot
    plt.xlabel('Patient ID')
    plt.ylabel('95th Percentile Hausdorff Distance (mm)')
    plt.title('95th Percentile Hausdorff Distance (HD95) per Organ Across Patients')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Create the directory if it doesn't exist
    save_dir = Path('results/segthor') / Path(bf)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "HD95_per_organ_across_patients.png"

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"95th Percentile Hausdorff Distance plot saved to {save_path}")


def plot_avg_hd_per_patient(patient_ids: List[str], all_results: np.ndarray, bf: str):
    """
    Plot a line graph of average Hausdorff distances (HD) across all organs per patient.

    Args:
        patient_ids (List[str]): List of patient identifiers.
        all_results (np.ndarray): Array of shape (num_patients, num_organs, 2) containing HD and HD95.
        bf (str): Base folder identifier for saving plots.
    """
    # Exclude background (class 0) and compute average HD per patient
    avg_hd = np.nanmean(all_results[:, 1:, 0], axis=1)  # Shape: (num_patients,)

    # Initialize plot
    plt.figure(figsize=(14, 8))
    plt.plot(patient_ids, avg_hd, label='Average HD', color='purple', linestyle='-')

    # Customize plot
    plt.xlabel('Patient ID')
    plt.ylabel('Average Hausdorff Distance (mm)')
    plt.title('Average Hausdorff Distance (HD) Across All Organs per Patient')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Create the directory if it doesn't exist
    save_dir = Path('results/segthor') / Path(bf)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "Average_HD_per_patient.png"

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Average Hausdorff Distance plot saved to {save_path}")


def plot_avg_hd95_per_patient(patient_ids: List[str], all_results: np.ndarray, bf: str):
    """
    Plot a line graph of average 95th Percentile Hausdorff distances (HD95) across all organs per patient.

    Args:
        patient_ids (List[str]): List of patient identifiers.
        all_results (np.ndarray): Array of shape (num_patients, num_organs, 2) containing HD and HD95.
        bf (str): Base folder identifier for saving plots.
    """
    # Exclude background (class 0) and compute average HD95 per patient
    avg_hd95 = np.nanmean(all_results[:, 1:, 1], axis=1)  # Shape: (num_patients,)

    # Initialize plot
    plt.figure(figsize=(14, 8))
    plt.plot(patient_ids, avg_hd95, label='Average HD95', color='orange', linestyle='--')

    # Customize plot
    plt.xlabel('Patient ID')
    plt.ylabel('Average 95th Percentile Hausdorff Distance (mm)')
    plt.title('Average 95th Percentile Hausdorff Distance (HD95) Across All Organs per Patient')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Create the directory if it doesn't exist
    save_dir = Path('results/segthor') / Path(bf)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "Average_HD95_per_patient.png"

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Average 95th Percentile Hausdorff Distance plot saved to {save_path}")


def run(args):
    """
    Main execution function to process images and generate plots.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Determine the base folder for ground truth based on the dataset and is_transformed_data flag
    if args.dataset.upper() == "SEGTHOR":
        gt_base_folder: str = "data/segthor_train/train"
    else:
        gt_base_folder: str = args.dataset  # Allow custom dataset paths

    # Extract ground truth images
    ground_truths: Dict[str, sitk.Image] = extract_files(
        gt_base_folder, "Patient_*/GT.nii.gz", r"Patient_(\d+)"
    )

    # Determine the predictions folder based on base_folder argument
    if args.base_folder:
        bf = args.base_folder
        pred_base_folder: str = f"volumes/segthor/{bf}/ce"
    else:
        bf = args.model
        pred_base_folder: str = f"volumes/segthor/{bf}/ce"

    # Extract prediction images
    predictions: Dict[str, sitk.Image] = extract_files(
        pred_base_folder, "Patient_*.nii.gz", r"Patient_(\d+)"
    )

    all_results = []
    patient_ids = []

    for patient_id, pred_image in predictions.items():
        if patient_id in ground_truths:
            gt_image = ground_truths[patient_id]
            print(f"\nPatient ID: {patient_id}")
            print_image_info(pred_image, "Prediction Before Resampling")
            print_image_info(gt_image, "Ground Truth")

            # Resample prediction to match ground truth if necessary
            if not (
                pred_image.GetSize() == gt_image.GetSize() and
                pred_image.GetSpacing() == gt_image.GetSpacing() and
                pred_image.GetOrigin() == gt_image.GetOrigin() and
                pred_image.GetDirection() == gt_image.GetDirection()
            ):
                print("Resampling prediction to match ground truth...")
                pred_image = resample_image(gt_image, pred_image, interpolator=sitk.sitkNearestNeighbor)
                print_image_info(pred_image, "Prediction After Resampling")

            # Proceed with distance calculations
            results = batch_hausdorff_distance(pred_image, gt_image)
            all_results.append(results)
            patient_ids.append(patient_id)
        else:
            print(f"Warning: No ground truth found for Patient ID {patient_id}")

    if not all_results:
        print("No matching predictions and ground truths found. Exiting.")
        return

    all_results = np.array(all_results)  # Shape: (num_patients, num_organs, 2)

    print("All Hausdorff distances (per patient):")
    print(all_results)

    # Sort patient_ids and all_results based on patient_ids for consistent plotting
    try:
        sorted_pairs = sorted(zip(patient_ids, all_results), key=lambda pair: int(pair[0]))
    except ValueError:
        # If patient IDs are not purely numeric, sort lexicographically
        sorted_pairs = sorted(zip(patient_ids, all_results), key=lambda pair: pair[0])
    patient_ids_sorted, all_results_sorted = zip(*sorted_pairs)
    patient_ids_sorted = list(patient_ids_sorted)
    all_results_sorted = np.array(all_results_sorted)

    # Print the sorted results
    print("Sorted Hausdorff distances (per patient):")
    print(all_results_sorted)

    # Now call the plotting functions
    plot_hd_per_organ(patient_ids_sorted, all_results_sorted, bf)
    plot_hd95_per_organ(patient_ids_sorted, all_results_sorted, bf)
    plot_avg_hd_per_patient(patient_ids_sorted, all_results_sorted, bf)
    plot_avg_hd95_per_patient(patient_ids_sorted, all_results_sorted, bf)


def get_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Plot Hausdorff distance per image")
    parser.add_argument(
        "--dataset",
        type=str,
        default="SEGTHOR",
        help="Dataset name. Default is 'SEGTHOR'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="UNet",
        help="Model name. Default is 'UNet'.",
    )
    parser.add_argument(
        "--base_folder",
        type=str,
        default=None,
        help="Base folder name for predictions. If provided, use this as the predictions folder; otherwise, use the model name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(get_args())
