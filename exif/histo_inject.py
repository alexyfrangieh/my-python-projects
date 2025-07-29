import os
import sys
import base64
import numpy as np
from PIL import Image
import subprocess
import json # Import the json module for serialization
import cv2 # Assuming you have OpenCV for histogram calculation

# You might need to install opencv-python if you haven't: pip install opencv-python
# Or adapt if you prefer to calculate histograms using only NumPy/Pillow

def compute_histogram_base64(image_path, bins=256):
    """
    Computes a 3-channel (BGR) histogram, applies a log transform,
    serializes it to JSON, encodes to UTF-8, and then Base64 encodes.
    Returns the Base64 string.
    """
    try:
        # Load image using PIL, convert to RGB, then to NumPy array for OpenCV
        img_pil = Image.open(image_path).convert('RGB')
        np_img = np.array(img_pil)

        # Convert RGB to BGR for OpenCV's calcHist function
        # If your plotting script assumes RGB order for the 768 bins, you might need
        # to adjust the slicing in plot_exif_histogram.py or change the order here.
        # Current plotting script assumes BGR order (Blue, Green, Red) from 0-255, 256-511, 512-767.
        img_cv = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        # Calculate histogram for each channel
        # density=True normalizes the histogram so the sum is 1.0
        hist_b = cv2.calcHist([img_cv], [0], None, [bins], [0,256])
        hist_g = cv2.calcHist([img_cv], [1], None, [bins], [0,256])
        hist_r = cv2.calcHist([img_cv], [2], None, [bins], [0,256])

        # Concatenate histograms into a single array (e.g., BGR order)
        combined_hist = np.concatenate([hist_b, hist_g, hist_r]).flatten()

        # Apply a logarithmic transformation (common for histogram visualization/feature extraction)
        # Adding a small epsilon (1e-6) to avoid log(0) for empty bins
        hist_transformed = np.log(combined_hist + 1e-6)

        # Convert the NumPy array of floats to a standard Python list of floats
        hist_list = hist_transformed.tolist()

        # Serialize the list of floats into a JSON string
        # This string will only contain ASCII characters (digits, '.', 'e', ',', '[', ']', '-')
        hist_json_string = json.dumps(hist_list)

        # Encode the JSON string into UTF-8 bytes
        # This is crucial to ensure the bytes are valid UTF-8 for later decoding
        hist_json_bytes = hist_json_string.encode('utf-8')

        # Base64 encode these UTF-8 bytes
        b64_encoded_bytes = base64.b64encode(hist_json_bytes)

        # Decode the Base64 bytes back to an ASCII string (Base64 output is always ASCII-safe)
        return b64_encoded_bytes.decode('ascii')

    except Exception as e:
        print(f"Error computing histogram for {image_path}: {e}")
        raise # Re-raise the exception to be caught by process_folder

def inject_histogram(image_path, b64_histogram):
    """
    Injects the Base64-encoded histogram string into the UserComment EXIF tag
    of the specified image using exiftool.
    """
    hist_tag = f"HISTO:b64:{b64_histogram}"

    # Use -m to allow minor errors, -fast to speed up, -n to disable print of values.
    # The crucial part is '-UserComment={hist_tag}' which sets the tag.
    # -overwrite_original directly modifies the file.
    command = [
        "exiftool",
        "-overwrite_original",
        f"-UserComment={hist_tag}",
        image_path
    ]

    # Run exiftool command silently
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"ExifTool error for {image_path}: {e}")
        raise # Re-raise for process_folder to catch

def process_folder(image_folder):
    """
    Walks through the specified folder and its subdirectories,
    computes and injects histograms for all supported image files.
    """
    if not os.path.isdir(image_folder):
        print(f"Error: Folder not found at '{image_folder}'")
        sys.exit(1)

    print(f"Starting histogram injection for images in: {image_folder}")
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')): # Focus on JPG for EXIF
                full_path = os.path.join(root, file)
                print(f"Processing: {full_path}")
                try:
                    # Use 256 bins to match the 768 total bins expected by the plotting script
                    b64_hist = compute_histogram_base64(full_path, bins=256)
                    inject_histogram(full_path, b64_hist)
                    print(f"  Successfully injected histogram.")
                except Exception as e:
                    print(f"  Failed for {full_path}: {e}")
    print(f"Finished processing images in: {image_folder}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histo_inject.py /path/to/images_folder")
        sys.exit(1)

    folder_to_process = sys.argv[1]
    process_folder(folder_to_process)
