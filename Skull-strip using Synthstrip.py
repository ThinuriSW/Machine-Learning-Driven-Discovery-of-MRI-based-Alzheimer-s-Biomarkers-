import os
import subprocess
import SimpleITK as sitk


base_dir = "/Users/thinurishehara/Desktop/OAS2_RAW_PART2"

def convert_hdr_to_nifti(hdr_path):
    try:
        image = sitk.ReadImage(hdr_path)
        nifti_path = hdr_path.replace(".hdr", ".nii.gz")
        sitk.WriteImage(image, nifti_path)
        print(f"Converted {hdr_path} to {nifti_path}")
        return nifti_path
    except Exception as e:
        print(f"Failed to convert {hdr_path}: {e}")
        return None

def run_synthstrip(nifti_path):

    folder = os.path.dirname(nifti_path)
    filename = os.path.basename(nifti_path)
    output_filename = filename.replace(".nii.gz", "_skullstripped.nii.gz")

    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{folder}:/data",
        "freesurfer/synthstrip",
        "-i", f"/data/{filename}",
        "-o", f"/data/{output_filename}"
    ]
    print(f"Running SynthStrip on {nifti_path}...")
    try:
        subprocess.run(docker_cmd, check=True)
        print(f"Skull-stripped image saved to: {os.path.join(folder, output_filename)}")
    except subprocess.CalledProcessError as e:
        print(f"Error running SynthStrip on {nifti_path}: {e}")

def main():

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".hdr"):
                hdr_path = os.path.join(root, file)
                nifti_path = convert_hdr_to_nifti(hdr_path)
                if nifti_path:
                    run_synthstrip(nifti_path)

if __name__ == "__main__":
    main()

