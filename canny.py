# import cv2
# import numpy as np

# def edge_detection(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, threshold1=50, threshold2=150)
#     return edges




# # ******************************************

import os
import cv2

def canny_edge_detection(input_dir, output_dir, threshold1=50, threshold2=150):
    """
    Apply Canny Edge Detection to all images in a directory.
    
    Parameters:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory to save processed images.
        threshold1 (int): First threshold for the hysteresis procedure in Canny.
        threshold2 (int): Second threshold for the hysteresis procedure in Canny.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # Read the image
        image = cv2.imread(input_path)

        if image is None:
            print(f"Skipping non-image file: {filename}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, threshold1, threshold2)

        # Save the processed image to the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, edges)

        print(f"Processed and saved: {output_path}")

# Set paths for your input and output directories
input_directory = r"Data/Damaged"
output_directory = r"Output"

# Run the function
canny_edge_detection(input_directory, output_directory)


