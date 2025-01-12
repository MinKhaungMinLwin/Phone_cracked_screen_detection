import os
import cv2
import albumentations as A
import uuid

# Input and output directories
input_dir = r"D:\python all collection\kopyae_2\crack_detect\dataset_main\dataset_2\train"  # Replace with your input directory
output_dir = r"D:\python all collection\kopyae_2\crack_detect\output_2"  # Replace with your output directory

# Number of augmented images to generate per input image
num_augmentations = 5

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Supported image formats
supported_formats = (".jpg", ".jpeg", ".png")

# Define the augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),               # Flip the image horizontally with 50% probability
    A.VerticalFlip(p=0.5),                 # Flip the image vertically with 50% probability
    A.RandomBrightnessContrast(p=0.2),    # Adjust brightness and contrast
    A.Rotate(limit=30, p=0.5),             # Random rotation within +/- 30 degrees
    A.GaussNoise(p=0.2),                   # Add Gaussian noise
    A.Blur(blur_limit=3, p=0.2),           # Apply blur
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),  # Slight shift, scale, and rotate
    A.Resize(256, 256)                     # Resize to 256x256
])

# Process each image in the input directory
for file_name in os.listdir(input_dir):
    if file_name.lower().endswith(supported_formats):  # Check if it's an image
        input_path = os.path.join(input_dir, file_name)

        try:
            # Read the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Could not read image: {file_name}")
                continue

            # Generate multiple augmentations
            for i in range(num_augmentations):
                augmented = transform(image=image)
                augmented_image = augmented["image"]

                # Create a unique filename
                unique_name = f"{os.path.splitext(file_name)[0]}_{uuid.uuid4().hex[:8]}.jpg"
                output_path = os.path.join(output_dir, unique_name)

                # Save the augmented image
                cv2.imwrite(output_path, augmented_image)
                print(f"Augmented: {unique_name}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

print("Data augmentation completed.")
