# plotting the class imbalances


# output file class_imbalances.py
# source folder with ground truths (images contain 5 labels): --source_scan_pattern 

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import os

def find_ground_truth_files(base_dir, gt_file_pattern):
    # Generate patient directories manually from 01 to 40
    patient_dirs = [f'data/segthor_train/train/Patient_{str(i).zfill(2)}' for i in range(1, 41)]
    
    return patient_dirs

def plot_class_percentage(class_counts, organ_names, output_file=None):
    total_voxels = np.sum(class_counts)  # Calculate the total number of voxels
    percentages = (class_counts / total_voxels) * 100  # Calculate percentage for each class

    plt.figure(figsize=(8, 6))
    plt.bar(organ_names[1:], percentages[1:], color=['green', 'yellow', 'red', 'blue'])
    plt.xlabel('Organs')
    plt.ylabel('Percentage of Voxels (%)')
    plt.title('Class Distribution in the Dataset (Percentage of Voxels)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Class percentage plot saved to {output_file}")
    else:
        plt.show()

# Function to count voxels for each class
def count_voxels_per_class(image_paths, num_classes):
    class_counts = np.zeros(num_classes)

    for image_path in image_paths:
        mask_sitk = sitk.ReadImage(image_path)  # Read the ground truth segmentation
        mask_np = sitk.GetArrayFromImage(mask_sitk)  # Convert to numpy array

        # Count voxels for each class label
        for c in range(num_classes):
            class_counts[c] += np.count_nonzero(mask_np == c)
    
    return class_counts

# Function to plot the class imbalances
def plot_class_imbalances(class_counts, organ_names, output_file=None):
    plt.figure(figsize=(8, 6))
    plt.bar(organ_names, class_counts, color=['black', 'green', 'yellow', 'red', 'blue'])
    plt.xlabel('Organs')
    plt.ylabel('Number of Voxels')
    plt.title('Class Distribution in the Dataset (Voxel Counts)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Class imbalance plot saved to {output_file}")
    else:
        plt.show()

    organ_counts = class_counts[1:]  # Exclude background count
    organ_names_without_bg = organ_names[1:]  # Exclude background name
    
    plt.figure(figsize=(8, 6))
    plt.bar(organ_names_without_bg, organ_counts, color=['green', 'yellow', 'red', 'blue'])
    plt.xlabel('Organs')
    plt.ylabel('Number of Voxels')
    plt.title('Class Distribution in the Dataset (Voxel Counts) - Excluding Background')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_imbalance_without_background.png')

# Main function
def main(source_scan_pattern, num_classes=5, output_plot='class_imbalance.png'):
    organ_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']

    base_dir = os.path.dirname(source_scan_pattern)
    gt_file_pattern = os.path.basename(source_scan_pattern)
    
    # Useglob to retrieve all matching image paths
    image_paths = find_ground_truth_files(base_dir,gt_file_pattern)
   
    # Count voxels for each class
    class_counts = count_voxels_per_class(image_paths, num_classes)

    # Plot and save the class imbalance chart
    plot_class_imbalances(class_counts, organ_names, output_plot)

    plot_class_percentage(class_counts, organ_names, output_file='class_percentage.png')

    print("class counts", class_counts)



if __name__ == '__main__':

   
    # Run the main function
    main(source_scan_pattern="ai4mi_project/data/segthor_train/train/Patient_{id}/GT.nii.gz", output_plot='class_imbalance.png')
