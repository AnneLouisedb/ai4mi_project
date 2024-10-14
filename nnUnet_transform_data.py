from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import argparse
import os
import random


def create_splits(train_patient_names, num_splits, num_val_cases):
    from collections import defaultdict
    patient_groups = defaultdict(list)
    for p in train_patient_names:
        base_id = p.split('_')[0]
        patient_groups[base_id].append(p)

    # Create a list of base patient IDs
    base_patient_ids = list(patient_groups.keys())
    random.shuffle(base_patient_ids)

    splits = []
    total_cases = len(base_patient_ids)
    fold_size = total_cases // num_splits

    for i in range(num_splits):
        val_base_ids = base_patient_ids[i * fold_size: (i + 1) * fold_size]
        val_cases = []
        train_cases = []

        for base_id in base_patient_ids:
            if base_id in val_base_ids:
                val_cases.extend(patient_groups[base_id])
            else:
                train_cases.extend(patient_groups[base_id])

        # Limit the number of validation cases if specified
        if num_val_cases and num_val_cases < len(val_cases):
            val_cases = val_cases[:num_val_cases]

        splits.append({'train': train_cases, 'val': val_cases})

    return splits



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SEGTHOR data directly to nnUNetv2 format with custom splits.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the SEGTHOR data directory.")
    parser.add_argument('--nnUNet_raw', type=str, required=True, help="Path to nnUNet raw data directory.")
    parser.add_argument('--nnUNet_preprocessed', type=str, required=True, help="Path to nnUNet preprocessed data directory.")
    parser.add_argument('--num_splits', type=int, default=5, help="Number of splits for cross-validation.")
    parser.add_argument('--num_val_cases', type=int, default=5, help="Number of validation cases per split.")
    parser.add_argument('--num_test_cases', type=int, default=5, help="Number of test cases.")
    parser.add_argument('--dataset_id', type=int, default= 1, help="Unique dataset ID.")
    parser.add_argument('--dataset_name', type=str, default = "SegTHOR", help="Dataset name (e.g., SegTHOR_Original).")
    args = parser.parse_args()

    # Get base directories from arguments
    base = args.data_dir
    nnUNet_raw = args.nnUNet_raw
    nnUNet_preprocessed = args.nnUNet_preprocessed

    task_id = args.dataset_id
    task_name = args.dataset_name

    # nnUNetv2 expects datasets to be named as DatasetXXX_Name
    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    all_patient_names = subfolders(join(base, "train"), join=False)
    all_patient_names.sort()

    # Randomly select test patients
    random.seed(1234)
    test_patient_names = random.sample(all_patient_names, args.num_test_cases)
    remaining_patient_names = [p for p in all_patient_names if p not in test_patient_names]

    # Process training and validation data
    train_patient_names = []
    for p in remaining_patient_names:
        curr = join(base, "train", p)
        label_file = join(curr, "GT.nii.gz")
        image_file = join(curr, p + ".nii.gz")
        shutil.copy(image_file, join(imagestr, p + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelstr, p + ".nii.gz"))
        train_patient_names.append(p)

    # Process test data
    for p in test_patient_names:
        curr = join(base, "train", p)
        image_file = join(curr, p + ".nii.gz")
        label_file = join(curr, "GT.nii.gz")
        # Copy images to imagesTs
        shutil.copy(image_file, join(imagests, p + "_0000.nii.gz"))
        # Copy labels to labelsTs
        shutil.copy(label_file, join(labelsts, p + ".nii.gz"))

    # Create the dataset JSON for nnUNetv2
    dataset_json = OrderedDict()

    # nnUNetv2 requires 'channel_names', 'labels', 'file_ending', and 'numTraining'
    dataset_json['channel_names'] = {
        "0": "CT",
    }

    # Labels are specified with names as keys and integer IDs as values
    dataset_json['labels'] = {
        "background": 0,
        "esophagus": 1,
        "heart": 2,
        "trachea": 3,
        "aorta": 4,
    }

    # Specify the number of training cases
    dataset_json['numTraining'] = len(train_patient_names)

    # Specify the file ending
    dataset_json['file_ending'] = ".nii.gz"

    # Save the dataset JSON
    save_json(dataset_json, os.path.join(out_base, "dataset.json"), sort_keys=False)

    print(f"{task_name} dataset has been converted to nnUNetv2 format at {out_base}")

    # Ensure preprocessing is done before creating splits
    preprocessed_folder = join(nnUNet_preprocessed, foldername)
    if not isdir(preprocessed_folder):
        print(f"Preprocessed data not found at {preprocessed_folder}. Please run nnUNetv2_plan_and_preprocess first.")
    else:
        # Create custom splits
        splits = create_splits(train_patient_names, args.num_splits, args.num_val_cases)

        # Path to the splits file in the preprocessed data directory
        splits_file = join(preprocessed_folder, 'splits_final.json')

        # Save the splits file
        save_json(splits, splits_file)

        print(f"Custom splits have been created and saved to {splits_file}")
