import os
import cv2
import uuid

def canny_edge_detection(input_dir, output_dir, threshold1=50, threshold2=150):
    """
    Apply Canny Edge Detection to all images in a directory and save them with unique filenames.

    Parameters:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory to save processed images.
        threshold1 (int): First threshold for the hysteresis procedure in Canny.
        threshold2 (int): Second threshold for the hysteresis procedure in Canny.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Supported image formats
    supported_formats = (".jpg", ".jpeg", ".png")

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):  # Check if it's an image
            input_path = os.path.join(input_dir, filename)

            # Read the image
            image = cv2.imread(input_path)

            if image is None:
                print(f"Skipping non-image file: {filename}")
                continue

            try:
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Apply Canny edge detection
                edges = cv2.Canny(gray, threshold1, threshold2)

                # Create a unique filename
                unique_name = f"{os.path.splitext(filename)[0]}_{uuid.uuid4().hex[:8]}.jpg"
                output_path = os.path.join(output_dir, unique_name)

                # Save the processed image
                cv2.imwrite(output_path, edges)

                print(f"Processed and saved: {output_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Set paths for your input and output directories
input_directory = r"D:\python all collection\kopyae_2\crack_detect\output_4_aug"  # Replace with your input directory
output_directory = r"D:\python all collection\kopyae_2\crack_detect\output_4_canny"  # Replace with your output directory

# Run the function
canny_edge_detection(input_directory, output_directory)
