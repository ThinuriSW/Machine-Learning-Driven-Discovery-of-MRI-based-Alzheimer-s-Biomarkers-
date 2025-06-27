import os
import nibabel as nib
import numpy as np
import csv
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew
import matplotlib.pyplot as plt

scan_dir = "/Users/thinurishehara/Desktop/OAS2_RAW_PART1"
mask_dir = "/Users/thinurishehara/Desktop/oas2_masks2"
os.makedirs(mask_dir, exist_ok=True)

summary = []

def compute_texture_features(image_2d, mask_2d, distances=[1], angles=[0], levels=256):

    image_2d = np.clip(image_2d, a_min=0, a_max=None)
    norm = (image_2d - np.min(image_2d)) / (np.ptp(image_2d) + 1e-8)
    image_uint8 = np.uint8(norm * 255)

    image_masked = image_uint8 * (mask_2d > 0)

    glcm = graycomatrix(image_masked, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))


    values = image_2d[mask_2d > 0]
    mean_val = np.mean(values)
    variance = np.var(values)
    skewness = skew(values)

    return {
        "GLCM_Contrast": contrast,
        "GLCM_Energy": energy,
        "GLCM_Homogeneity": homogeneity,
        "GLCM_Correlation": correlation,
        "GLCM_Entropy": entropy,
        "Mean_Intensity": mean_val,
        "Variance_Intensity": variance,
        "Skewness_Intensity": skewness
    }

for root, dirs, files in os.walk(scan_dir):
    for file in files:
        if file.endswith(".nii.gz") and "skullstripped" in file:
            image_path = os.path.join(root, file)

            patient_folder = os.path.basename(os.path.dirname(root))
            file_base = os.path.splitext(os.path.splitext(file)[0])[0]
            subject_id = f"{patient_folder}_{file_base}"

            mask_path = os.path.join(mask_dir, f"{subject_id}_mask.nii.gz")
            overlay_path = os.path.join(mask_dir, f"{subject_id}_overlay.png")

            try:
                img_nifti = nib.load(image_path)
                img_data = img_nifti.get_fdata()
                mask_nifti = nib.load(mask_path)
                mask_data = mask_nifti.get_fdata()

                voxel_dims = img_nifti.header.get_zooms()[:3]
                voxel_volume_mm3 = np.prod(voxel_dims)
                brain_volume_mm3 = np.sum(mask_data) * voxel_volume_mm3
                brain_volume_cm3 = brain_volume_mm3 / 1000

                z = img_data.shape[2] // 2
                slice_img = img_data[:, :, z]
                slice_mask = mask_data[:, :, z]

                features = compute_texture_features(slice_img, slice_mask)
                features.update({
                    "Subject_ID": subject_id,
                    "Filename": file,
                    "Volume_mm3": brain_volume_mm3,
                    "Volume_cm3": brain_volume_cm3
                })

                plt.imshow(slice_img, cmap='gray')
                plt.imshow(slice_mask, alpha=0.3, cmap='Reds')
                plt.title(f"{subject_id} Mask Overlay")
                plt.axis('off')
                plt.savefig(overlay_path, bbox_inches='tight')
                plt.close()

                summary.append(features)

csv_path = os.path.join(mask_dir, "brain_texture_summary.csv")
with open(csv_path, "w", newline="") as f:
    fieldnames = [
        "Subject_ID",
        "Filename",
        "Volume_mm3",
        "Volume_cm3",
        "GLCM_Contrast",
        "GLCM_Energy",
        "GLCM_Homogeneity",
        "GLCM_Correlation",
        "GLCM_Entropy",
        "Mean_Intensity",
        "Variance_Intensity",
        "Skewness_Intensity"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary)
