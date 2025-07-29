import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")

# Create a simple black image (100x100 pixels, 3 color channels)
img = np.zeros((100, 100, 3), dtype=np.uint8)
print(f"Created a black image of shape: {img.shape}")

# You won't see a window on a server, but this tests import and basic function
# cv2.imwrite("black_image.png", img) # Uncomment to save an image file
print("OpenCV test complete.")
