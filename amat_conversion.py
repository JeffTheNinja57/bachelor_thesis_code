import os
import numpy as np
from PIL import Image
import glob


def create_amat_from_images(input_dir, output_file, img_width=256, img_height=64):
    """
    Convert a directory of PNG images to a single AMAT file.

    This function processes all PNG images in the specified directory and converts them
    into a single AMAT file format. Each row in the AMAT file contains a flattened image
    (with pixel values normalized to [0,1]) followed by its class label.

    The class label is extracted from the filename (first number before underscore).
    The function handles both RGB and grayscale images, and automatically detects
    the actual dimensions of the images.

    :param input_dir: Directory containing PNG images
    :type input_dir: str
    :param output_file: Path to the output AMAT file
    :type output_file: str
    :param img_width: Expected width of input images (used only for empty dataset creation)
    :type img_width: int
    :param img_height: Expected height of input images (used only for empty dataset creation)
    :type img_height: int
    :return: None
    :rtype: None
    """
    # Get all PNG files in the directory
    image_files = glob.glob(os.path.join(input_dir, "*.png"))

    if len(image_files) == 0:
        print(f"No PNG files found in {input_dir}")
        # Create an empty file with the right structure
        dataset = np.zeros((0, img_width * img_height * 3 + 1))
        np.savetxt(output_file, dataset)
        print(f"Dataset saved with shape {dataset.shape}")
        return

    # Check the first image to get dimensions
    sample_img = Image.open(image_files[0])
    actual_width, actual_height = sample_img.size
    channels = len(np.array(sample_img).shape)
    if channels == 3:
        # RGB image
        num_channels = 3
    else:
        # Grayscale image
        num_channels = 1

    print(f"Image dimensions: {actual_width}x{actual_height} with {num_channels} channels")

    # Create an array to hold all the flattened images and labels
    num_pixels = actual_width * actual_height * num_channels
    dataset = np.zeros((len(image_files), num_pixels + 1))

    print(f"Processing {len(image_files)} images from {input_dir}...")

    for i, img_path in enumerate(image_files):
        if i % 100 == 0:
            print(f"Processing image {i + 1}/{len(image_files)}")

        # Extract the class label from the filename (first number before underscore)
        filename = os.path.basename(img_path)
        label = str(filename.split('_')[2])

        if label == "pull":
            label = 0
        elif label == "push":
            label = 1
        elif label == "right":
            label = 2
        elif label == "left":
            label = 3

        # Open and convert the image to numpy array
        img = Image.open(img_path)
        img_array = np.array(img)

        # Ensure consistent shape for grayscale images
        if len(img_array.shape) == 2:
            # Grayscale image, add channel dimension
            img_array = img_array.reshape(img_array.shape[0], img_array.shape[1], 1)

        # Normalize pixel values to [0, 1]
        img_array = img_array / 255.0

        # Flatten the image
        flattened_img = img_array.flatten()

        # Store the flattened image and label
        dataset[i, :-1] = flattened_img
        dataset[i, -1] = label

    # Save the dataset to AMAT file
    print(f"Saving dataset to {output_file}...")
    np.savetxt(output_file, dataset)
    print(f"Dataset saved with shape {dataset.shape}")


# Process the 'early' dataset
early_train_dir = "data/early/train"
early_test_dir = "data/early/test"
early_val_dir = "data/early/val"

# Create output directory if it doesn't exist
output_dir = "datasets/early-dataset"
os.makedirs(output_dir, exist_ok=True)

# Convert early dataset
create_amat_from_images(early_train_dir, os.path.join(output_dir, "early_train.amat"))
create_amat_from_images(early_test_dir, os.path.join(output_dir, "early_test.amat"))

# Optionally, also process validation data
create_amat_from_images(early_val_dir, os.path.join(output_dir, "early_val.amat"))

# Process late_color dataset (similar steps)
late_color_train_dir = "data/late_color/train"
late_color_test_dir = "data/late_color/test"
late_color_val_dir = "data/late_color/val"

output_dir = "datasets/late-color-dataset"
os.makedirs(output_dir, exist_ok=True)

create_amat_from_images(late_color_train_dir, os.path.join(output_dir, "late_color_train.amat"))
create_amat_from_images(late_color_test_dir, os.path.join(output_dir, "late_color_test.amat"))
create_amat_from_images(late_color_val_dir, os.path.join(output_dir, "late_color_val.amat"))

# Process late_depth dataset (similar steps)
late_depth_train_dir = "data/late_depth/train"
late_depth_test_dir = "data/late_depth/test"
late_depth_val_dir = "data/late_depth/val"

output_dir = "datasets/late-depth-dataset"
os.makedirs(output_dir, exist_ok=True)

create_amat_from_images(late_depth_train_dir, os.path.join(output_dir, "late_depth_train.amat"))
create_amat_from_images(late_depth_test_dir, os.path.join(output_dir, "late_depth_test.amat"))
create_amat_from_images(late_depth_val_dir, os.path.join(output_dir, "late_depth_val.amat"))
