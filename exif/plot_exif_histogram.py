import argparse
import json
import base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS
import piexif
import os

def extract_and_plot_histogram(image_path):
    """
    Extracts a Base64-encoded histogram from the UserComment EXIF tag
    and plots it in a Lightroom-style format.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    try:
        img_pil = Image.open(image_path)
    except IOError:
        print(f"Error: Could not open image file '{image_path}'. Is it a valid image?")
        return

    exif_data = None
    if "exif" in img_pil.info:
        try:
            exif_data = piexif.load(img_pil.info["exif"])
        except piexif.InvalidImageDataError:
            print(f"Warning: Could not parse EXIF data for '{image_path}'. EXIF might be corrupted or malformed.")
            exif_data = None
    else:
        print(f"No EXIF data found in '{image_path}'.")
        print("Please ensure your image has the histogram embedded in its EXIF 'UserComment' tag.")
        return

    histogram_data_string = None
    # Find the UserComment tag ID (37510)
    user_comment_tag_id = None
    for tag, value in TAGS.items():
        if value == "UserComment":
            user_comment_tag_id = tag
            break

    if exif_data and "Exif" in exif_data and user_comment_tag_id in exif_data["Exif"]:
        try:
            histogram_data_bytes = exif_data["Exif"][user_comment_tag_id]

            # --- MODIFICATION START ---
            # Try decoding with common EXIF UserComment prefixes removed
            decoded_candidate = None

            # Common prefixes for UserComment
            prefixes_to_try = [
                b"ASCII\0\0\0",  # Standard ASCII prefix (8 bytes)
                b"UNICODE\0",   # Standard UNICODE prefix (8 bytes)
                b"JIS\0\0\0\0\0",# Standard JIS prefix (8 bytes)
                b"",            # No prefix (direct string)
            ]

            for prefix in prefixes_to_try:
                if histogram_data_bytes.startswith(prefix):
                    content_bytes = histogram_data_bytes[len(prefix):]
                    try:
                        # Try decoding as UTF-8 first, as that's what we expect from JSON
                        decoded_candidate = content_bytes.decode('utf-8').strip('\0')
                        if decoded_candidate.startswith("HISTO:b64:"):
                            histogram_data_string = decoded_candidate
                            print(f"UserComment decoded successfully with prefix '{prefix.decode('ascii', errors='ignore')}' and UTF-8.")
                            break
                    except UnicodeDecodeError:
                        # If UTF-8 fails, try latin-1, as exiftool often shows it this way
                        try:
                            decoded_candidate = content_bytes.decode('latin-1').strip('\0')
                            if decoded_candidate.startswith("HISTO:b64:"):
                                histogram_data_string = decoded_candidate
                                print(f"UserComment decoded successfully with prefix '{prefix.decode('ascii', errors='ignore')}' and Latin-1.")
                                break
                        except Exception as e_latin1:
                            # print(f"Debug: Failed to decode with latin-1 after stripping '{prefix.decode('ascii', errors='ignore')}': {e_latin1}")
                            pass
                # else:
                    # print(f"Debug: Data does not start with prefix '{prefix.decode('ascii', errors='ignore')}'")


            if histogram_data_string is None:
                # Fallback if no specific prefix worked, try general decoding
                try:
                    histogram_data_string = histogram_data_bytes.decode('utf-8').strip('\0')
                    if not histogram_data_string.startswith("HISTO:b64:"):
                        histogram_data_string = histogram_data_bytes.decode('latin-1').strip('\0')
                        if not histogram_data_string.startswith("HISTO:b64:"):
                             histogram_data_string = None # Still not found
                except Exception as e_fallback:
                    print(f"Debug: Fallback decoding failed: {e_fallback}")
                    histogram_data_string = None

            # --- MODIFICATION END ---

        except Exception as e:
            print(f"Error accessing or decoding UserComment EXIF tag: {e}")
            histogram_data_string = None
    else:
        print(f"No 'UserComment' EXIF tag found in '{image_path}' or no EXIF data parsed.")
        print("Please ensure the histogram is embedded in 'UserComment' (tag ID 37510).")
        return

    if not histogram_data_string:
        print("Could not retrieve histogram data from EXIF. Exiting.")
        return

    # Check for and extract the Base64 portion
    if histogram_data_string.startswith("HISTO:b64:"):
        try:
            b64_encoded_data = histogram_data_string.split(":", 2)[2]
            decoded_bytes = base64.b64decode(b64_encoded_data)
            decoded_json_string = decoded_bytes.decode('utf-8')
            histogram_values = json.loads(decoded_json_string)
            histogram_array = np.array(histogram_values)
        except Exception as e:
            print(f"Error decoding or parsing histogram data: {e}")
            print("Ensure your embedded histogram is 'HISTO:b64:<base64_json_array>'.")
            return
    else:
        # This case should ideally not be reached if the above logic worked
        print("UserComment content does not match expected 'HISTO:b64:' format after decoding attempts.")
        print("Ensure your embedded histogram is 'HISTO:b64:<base64_json_array>'.")
        return

    # --- Plotting the histogram ---
    blue_hist, green_hist, red_hist = None, None, None
    luminance_hist = None

    if len(histogram_array) == 768:
        # Assuming BGR order if calculated by OpenCV and then stored R,G,B order
        # If your histogram was calculated as RGB, adjust these slices accordingly
        blue_hist = histogram_array[0:256]
        green_hist = histogram_array[256:512]
        red_hist = histogram_array[512:768]
        luminance_hist = (red_hist + green_hist + blue_hist) / 3.0 # Simple average for luminance
    elif len(histogram_array) == 256:
        # Assume it's a grayscale/luminance histogram directly
        luminance_hist = histogram_array
        print("Interpreting as a 256-bin grayscale histogram.")
    else:
        print(f"Warning: Unexpected number of histogram bins ({len(histogram_array)}). Cannot plot effectively.")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.title(f'Image Histogram for: {os.path.basename(image_path)}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency (Relative)')
    plt.xlim([0, 255])
    plt.xticks([0, 64, 128, 192, 255], ['Blacks', 'Shadows', 'Midtones', 'Highlights', 'Whites'])

    bins = np.arange(256) # 0 to 255

    # Plot individual color channels as filled areas with transparency
    if blue_hist is not None:
        plt.fill_between(bins, blue_hist.flatten(), color='blue', alpha=0.3)
        plt.fill_between(bins, green_hist.flatten(), color='green', alpha=0.3)
        plt.fill_between(bins, red_hist.flatten(), color='red', alpha=0.3)

    # Plot the luminance histogram as a line on top
    if luminance_hist is not None:
        plt.plot(bins, luminance_hist.flatten(), color='gray', linewidth=1.5, alpha=0.9, label='Luminance')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts a Base64-encoded histogram from EXIF UserComment and plots it.")
    parser.add_argument("image_file", type=str, help="Path to the image file (e.g., photo.jpg)")

    args = parser.parse_args()

    extract_and_plot_histogram(args.image_file)
