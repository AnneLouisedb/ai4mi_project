import nibabel as nib
import numpy as np

# Load the ground truth and fake segmentation masks for a specific patient
ground_truth_mask = nib.load("data\\segthor_train\\train\\Patient_27\\GT2.nii.gz")
fake_mask = nib.load("data\\segthor_train\\train\\Patient_27\\GT.nii.gz")

# Convert the masks to NumPy arrays
ground_truth_array = np.array(ground_truth_mask.dataobj)
fake_array = np.array(fake_mask.dataobj)

# Create binary masks for the heart (assuming 2 represents the heart)
heart_mask_real = (ground_truth_array == 2)
heart_mask_fake = (fake_array == 2)

# Function to compute the centroid of the heart in the mask
def calculate_centroid(mask):
    coords = np.argwhere(mask == 1)  # Get coordinates where the heart is present
    centroid = np.mean(coords, axis=0)  # Calculate the mean position (centroid)
    return centroid

# Calculate the centroids for real and fake heart masks
centroid_real = calculate_centroid(heart_mask_real)
centroid_fake = calculate_centroid(heart_mask_fake)

# Calculate the translation vector to align the hearts
translation_vector = centroid_real - centroid_fake
print(f"Translation vector: {translation_vector}")

# Function to translate the heart mask based on the translation vector
def shift_mask(mask, translation):
    coords = np.argwhere(mask == 1)  # Find indices of the heart (1)
    
    # Apply translation to these indices
    shifted_coords = coords + translation
    
    # Create a new array for the shifted heart mask
    shifted_mask = np.zeros_like(mask)
    
    # Set the shifted positions to 1 (heart)
    for coord in shifted_coords.astype(int):
        shifted_mask[tuple(coord)] = 1
    
    return shifted_mask

# Function to save a NumPy array as a NIfTI file
def save_mask_as_nifti(array, output_filename, reference_mask):
    nifti_image = nib.Nifti1Image(array, affine=reference_mask.affine, header=reference_mask.header)
    nib.save(nifti_image, output_filename)

# Iterate over patient numbers from 1 to 39
for patient_num in range(1, 40):
    # Format patient number to have leading zero if necessary
    if patient_num < 10:
        patient_num = "0" + str(patient_num)
    
    print(f"Processing patient {patient_num}")
    
    # Load the fake segmentation mask for the current patient
    patient_fake_mask = nib.load(f"data\\segthor_train\\train\\Patient_{patient_num}\\GT.nii.gz")
    patient_fake_array = np.array(patient_fake_mask.dataobj)

    # Save a copy of the fake mask and remove the heart (value 2)
    preserved_mask = np.copy(patient_fake_array)
    preserved_mask[preserved_mask == 2] = 0

    # Create a binary mask for the heart in the fake image
    heart_mask_fake = (patient_fake_array == 2)

    # Shift the heart mask
    adjusted_heart_mask = shift_mask(heart_mask_fake, translation_vector)

    # Re-add the other organs to the adjusted heart mask
    preserved_mask[adjusted_heart_mask == 1] = 2

    # Save the updated segmentation mask
    save_mask_as_nifti(preserved_mask, f"data\\segthor_train\\train\\Patient_{patient_num}\\GT2.nii.gz", patient_fake_mask)

print("Processing complete")
