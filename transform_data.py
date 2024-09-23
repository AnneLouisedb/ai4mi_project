import SimpleITK as sitk
import os
import shutil
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
from scipy import ndimage
from pathlib import Path

replace_index = 2

def register_images(fixed_img_file, moving_img_file, output_transform_file, replace_index=None):
    """
    Register two images and save the transform.
    """
    # Read the images
    fixed_img = sitk.ReadImage(fixed_img_file, sitk.sitkFloat32)
    moving_img = sitk.ReadImage(moving_img_file, sitk.sitkFloat32)
    
    if replace_index:
        fixed_img = sitk.GetArrayFromImage(fixed_img)
        fixed_img = (fixed_img == replace_index).astype(np.float32)
        fixed_img = sitk.GetImageFromArray(fixed_img)
        
        moving_img = sitk.GetArrayFromImage(moving_img)
        moving_img = (moving_img == replace_index).astype(np.float32)
        moving_img = sitk.GetImageFromArray(moving_img)

    # Setup the registration
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.02)
    registration.SetInterpolator(sitk.sitkNearestNeighbor)
    registration.SetOptimizerAsGradientDescent(learningRate=0.8, numberOfIterations=200, convergenceMinimumValue=1e-7, convergenceWindowSize=1)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(sitk.CenteredTransformInitializer(fixed_img, moving_img, sitk.AffineTransform(3), True))
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Run the registration
    transform = registration.Execute(fixed_img, moving_img)
    
    # Save the transform
    sitk.WriteTransform(transform, output_transform_file)


# Step 2: Apply affine transformation using SciPy
def apply_affine_transform(moving_image):
    # transform_parameters = list(map(float, transform_map.GetParameterMap(1).get('TransformParameters') + transform_map.GetParameterMap(0).get('TransformParameters')))

    # Hardcoded transform parameters from the raw transform file
    transform_parameters = (1,0,0,0,1,0,0,0,1,-66.58184178354128,-38.56866728429671,-14.999585668903961)
    affine_matrix = np.array([
        [transform_parameters[0], transform_parameters[1], transform_parameters[2], transform_parameters[9]],
        [transform_parameters[3], transform_parameters[4], transform_parameters[5], transform_parameters[10]],
        [transform_parameters[6], transform_parameters[7], transform_parameters[8], transform_parameters[11]],
        [0, 0, 0, 1]
    ])

    rotation_scale = affine_matrix[:3, :3]
    translation = affine_matrix[:3, 3]

    moving_image_data = np.array(moving_image)
    transformed_image_data = affine_transform(moving_image_data, rotation_scale, offset=translation, order=0)

    object_center = ndimage.center_of_mass(transformed_image_data)
    volume_center = np.array(transformed_image_data.shape) // 2
    # shift = volume_center - np.array(object_center)
    # Hardcoded shift
    shift = np.array([-30.68731694,  21.74786194,  20.2083591 ])

    shifted_volume = ndimage.shift(transformed_image_data, shift, order=0)

    angle = -21  # Rotation angle in degrees
    axes = (0,1)  # Rotation in the (x,y) plane

    # Rotate the 3D volume
    rotated_volume = ndimage.rotate(shifted_volume, angle, axes=axes, reshape=False, order=0)

    # Shift the rotated volume back to the original position
    transformed_image_data = ndimage.shift(rotated_volume, -shift, order=0)

    return transformed_image_data

# Step 3: Save the result using Nibabel
def save_image_with_nibabel(transformed_image_data, reference_image_path, output_image_path, replace_index=None):
    reference_image = nib.load(reference_image_path)
    reference_affine = reference_image.affine
    reference_header = reference_image.header
    
    if replace_index:
        reference_image_data = reference_image.get_fdata()
        reference_image_data[reference_image_data == replace_index] = 0
        reference_image_data[transformed_image_data == 1] = replace_index
        transformed_image_data = reference_image_data

    transformed_image_nifti = nib.Nifti1Image(transformed_image_data.astype(np.uint8), reference_affine, reference_header)
    nib.save(transformed_image_nifti, output_image_path)

def apply_transform_to_segmentation(input_seg_file, transform_file, output_file):
    """
    Apply the BSpline transform to a .nii.gz segmentation and save the result.
    """
    # Read the input segmentation
    input_seg = sitk.ReadImage(input_seg_file)
    
    # Read the transform
    transform = sitk.ReadTransform(transform_file)
    
    # Setup the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(input_seg)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # NearestNeighbor is usually used for segmentation
    
    # Apply the transform
    transformed_seg = resampler.Execute(input_seg)
    
    # Save the transformed segmentation
    sitk.WriteImage(transformed_seg, output_file)

def main():
    base_folder = 'data/segthor_train/train'
    transform_file = 'data/Slicer BSpline Transform.h5'
    
    # Loop through all patient folders
    for patient_id in range(1, 41): #range(1, 41):  # Assuming 40 patients
        patient_folder = f'Patient_{patient_id:02d}'
        input_seg_file = os.path.join(base_folder, patient_folder, 'GT.nii.gz')
        moving_image = nib.load(input_seg_file).get_fdata()
        if replace_index:
            moving_image = (moving_image  == replace_index).astype(np.float32)
        
        print("Applying affine transformation...")
        transformed_image_data = apply_affine_transform(moving_image)
        print("Saving the transformed image...")

        output_image_path = os.path.join(base_folder, patient_folder, 'transformed_GT_manual.nii.gz')

        input_seg_file_2 = os.path.join(base_folder, patient_folder, 'GT.nii.gz')

        save_image_with_nibabel(transformed_image_data.round(), input_seg_file_2, output_image_path, replace_index=replace_index)
        print(f"Saved to {output_image_path}")



if __name__ == '__main__':
    main()






