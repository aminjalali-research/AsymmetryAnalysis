#!/bin/bash

# Resample parcellation from native space to MNI space
# This matches the parcellation dimensions to the voxel-based analysis (72x87x72)

echo "=================================="
echo "Resample Parcellation to MNI Space"
echo "=================================="
echo ""

# Process each patient
for PATIENT in P013 P014 P015 P016 P017 P018 P019 P020 P021 P022 P023 P024 P025 P026 P027
do
    echo "Processing ${PATIENT}..."
    
    INPUT_PARC="Dataset MNI/${PATIENT}/aparc+aseg.nii.gz"
    MNI_REF="Dataset MNI/${PATIENT}/perfusion_calib.nii.gz"
    OUTPUT_PARC="Dataset MNI/${PATIENT}/aparc+aseg_mni_resampled.nii.gz"
    
    # Check if files exist
    if [ ! -f "${INPUT_PARC}" ]; then
        echo "  ⚠️  Parcellation not found, skipping..."
        continue
    fi
    
    if [ ! -f "${MNI_REF}" ]; then
        echo "  ⚠️  MNI T1w not found, skipping..."
        continue
    fi
    
    # Resample using Python/nibabel to match exact dimensions
    echo "  → Resampling to MNI space..."
    python3 << PYTHON_EOF
import nibabel as nib
from scipy.ndimage import affine_transform
import numpy as np

# Load parcellation and MNI reference
parc_img = nib.load("${INPUT_PARC}")
mni_img = nib.load("${MNI_REF}")

parc_data = parc_img.get_fdata()
target_shape = mni_img.shape
target_affine = mni_img.affine

# Resample using nearest neighbor
from scipy.ndimage import zoom
zoom_factors = [target_shape[i] / parc_data.shape[i] for i in range(3)]
resampled_data = zoom(parc_data, zoom_factors, order=0)  # order=0 = nearest neighbor

# Save resampled parcellation
resampled_img = nib.Nifti1Image(resampled_data.astype(np.int16), target_affine)
nib.save(resampled_img, "${OUTPUT_PARC}")
print(f"  Resampled from {parc_data.shape} to {resampled_data.shape}")
PYTHON_EOF
    
    if [ $? -eq 0 ]; then
        echo "  ✅ ${PATIENT} resampled successfully"
    else
        echo "  ❌ ${PATIENT} failed"
    fi
    
done

echo ""
echo "=================================="
echo "✅ Resampling complete!"
echo "=================================="
echo ""
echo "Output files: Dataset MNI/{patient}/aparc+aseg_mni_resampled.nii.gz"
echo ""
echo "Next step: Run python voxel_roi_extraction.py"
