import os
from rembg import remove
from PIL import Image

# Input and output directory paths
input_dir = r"D:\python all collection\kopyae_2\crack_detect\output_2"
output_dir = r"D:\python all collection\kopyae_2\crack_detect\output_rembg_5550"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Supported image formats
supported_formats = (".jpg", ".jpeg", ".png")

# Process each file in the input directory
for file_name in os.listdir(input_dir):
    if file_name.lower().endswith(supported_formats):  # Check if it's an image file
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        try:
            # Open the image
            with open(input_path, "rb") as input_file:
                input_image = input_file.read()

            # Remove the background
            output_image = remove(input_image)

            # Save the output image
            with open(output_path, "wb") as output_file:
                output_file.write(output_image)

            print(f"Processed: {file_name}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

print("Background removal completed.")
