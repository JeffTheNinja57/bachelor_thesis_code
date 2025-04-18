#!/usr/bin/env python3
"""
concat_images.py - Image Concatenation for Action Recognition Dataset

This script processes an action recognition dataset by concatenating images in two ways:
1. Late Fusion: Horizontally concatenate pairs of images (color-color and depth-depth)
2. Early Fusion: Horizontally concatenate all four images (color-color-depth-depth)

The script traverses a specific directory structure to find the images, resizes them to 64x64 if needed,
and saves the resulting images in specific output directories.

Usage:
    python concat_images.py

Requirements:
    - PIL (Pillow) for image processing
    - os, glob for directory traversal
    - logging for progress tracking
"""

import os
import glob
import logging
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
TARGET_SIZE = (64, 64)  # Target size for each individual image
DATASET_PATH = '../datasets/action_recognition_dataset'
OUTPUT_DIRS = {
    'late_color': 'data/late_color',
    'late_depth': 'data/late_depth',
    'early': 'data/early'
}
VALID_CAMERAS = ['color', 'depthcolormap']  # Only process these camera views


def create_output_directories():
    """Create output directories if they don't exist."""
    for dir_path in OUTPUT_DIRS.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created output directory: {dir_path}")
        else:
            logger.info(f"Output directory already exists: {dir_path}")


def load_and_resize_image(image_path):
    """
    Load an image from disk and resize it to the target size.

    Args:
        image_path (str): Path to the image file

    Returns:
        PIL.Image: Resized image
    """
    try:
        img = Image.open(image_path)

        # Resize if not already the target size
        if img.size != TARGET_SIZE:
            img = img.resize(TARGET_SIZE, Image.LANCZOS)

        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def horizontal_concat_images(images):
    """
    Horizontally concatenate a list of images.

    Args:
        images (list): List of PIL.Image objects

    Returns:
        PIL.Image: Concatenated image
    """
    if not images or None in images:
        return None

    # Calculate dimensions for the concatenated image
    # Width = sum of all image widths, Height = height of first image
    total_width = sum(img.width for img in images)
    height = images[0].height

    # Create a new blank image with the calculated dimensions
    concat_img = Image.new('RGB', (total_width, height))

    # Paste each image side by side
    x_offset = 0
    for img in images:
        concat_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return concat_img


def process_sample(object_idx, object_name, tool, action, init_color_path, effect_color_path,
                  init_depth_path, effect_depth_path):
    """
    Process a single sample (4 images) and create late and early fusion images.

    Args:
        object_idx (int): Object index
        object_name (str): Object name
        tool (str): Tool name
        action (str): Action name
        init_color_path (str): Path to initial color image
        effect_color_path (str): Path to effect color image
        init_depth_path (str): Path to initial depth image
        effect_depth_path (str): Path to effect depth image
    """
    # Load and resize all four images
    init_color_img = load_and_resize_image(init_color_path)
    effect_color_img = load_and_resize_image(effect_color_path)
    init_depth_img = load_and_resize_image(init_depth_path)
    effect_depth_img = load_and_resize_image(effect_depth_path)

    # Check if all images were loaded successfully
    if None in [init_color_img, effect_color_img, init_depth_img, effect_depth_img]:
        logger.error(f"Failed to load all images for {object_idx}_{tool}_{action}")
        return

    # Generate base filename for output
    base_filename = f"{object_idx}_{tool}_{action}"

    # Late Fusion: Color (init_color + effect_color)
    late_color_img = horizontal_concat_images([init_color_img, effect_color_img])
    if late_color_img:
        late_color_path = os.path.join(OUTPUT_DIRS['late_color'], f"{base_filename}_color.png")
        late_color_img.save(late_color_path)

    # Late Fusion: Depth (init_depth + effect_depth)
    late_depth_img = horizontal_concat_images([init_depth_img, effect_depth_img])
    if late_depth_img:
        late_depth_path = os.path.join(OUTPUT_DIRS['late_depth'], f"{base_filename}_depth.png")
        late_depth_img.save(late_depth_path)

    # Early Fusion: All four images (init_color + effect_color + init_depth + effect_depth)
    early_fusion_img = horizontal_concat_images([init_color_img, effect_color_img, init_depth_img, effect_depth_img])
    if early_fusion_img:
        early_fusion_path = os.path.join(OUTPUT_DIRS['early'], f"{base_filename}_early.png")
        early_fusion_img.save(early_fusion_path)


def process_dataset():
    """
    Process the entire dataset by traversing the directory structure.

    Directory structure:
    data/action_recognition_dataset/<object_idx>_<objectName>/<tool>/<action>/<camera>/<filename.png>
    """
    # Create output directories
    create_output_directories()

    # Get all object directories
    object_dirs = [d for d in os.listdir(DATASET_PATH)
                  if os.path.isdir(os.path.join(DATASET_PATH, d)) and '_' in d]

    # Process each object
    for object_dir in sorted(object_dirs):
        # Extract object index and name
        parts = object_dir.split('_', 1)
        if len(parts) != 2 or not parts[0].isdigit():
            logger.warning(f"Skipping directory with invalid format: {object_dir}")
            continue

        object_idx = int(parts[0])
        object_name = parts[1]

        # Skip if object index is not in range 0-19
        if object_idx < 0 or object_idx > 19:
            logger.warning(f"Skipping object with index out of range (0-19): {object_idx}")
            continue

        logger.info(f"Processing object {object_idx}: {object_name}")

        # Get all tool directories for this object
        object_path = os.path.join(DATASET_PATH, object_dir)
        tool_dirs = [d for d in os.listdir(object_path)
                    if os.path.isdir(os.path.join(object_path, d))]

        # Process each tool
        for tool in sorted(tool_dirs):
            logger.info(f"  Processing tool: {tool}")

            # Get all action directories for this tool
            tool_path = os.path.join(object_path, tool)
            action_dirs = [d for d in os.listdir(tool_path)
                          if os.path.isdir(os.path.join(tool_path, d))]

            # Process each action
            for action in sorted(action_dirs):
                logger.info(f"    Processing action: {action}")

                # Base path for this action
                action_path = os.path.join(tool_path, action)

                # Check if both required camera views exist
                if not all(os.path.isdir(os.path.join(action_path, camera)) for camera in VALID_CAMERAS):
                    logger.warning(f"      Skipping action missing required camera views: {action}")
                    continue

                # Get all image files for color view
                color_path = os.path.join(action_path, 'color')
                init_color_files = sorted(glob.glob(os.path.join(color_path, 'init_color_*.png')))
                effect_color_files = sorted(glob.glob(os.path.join(color_path, 'effect_color_*.png')))

                # Get all image files for depth view
                depth_path = os.path.join(action_path, 'depthcolormap')
                init_depth_files = sorted(glob.glob(os.path.join(depth_path, 'init_depthcolormap_*.png')))
                effect_depth_files = sorted(glob.glob(os.path.join(depth_path, 'effect_depthcolormap_*.png')))

                # Process each set of images (matching by index)
                for i in range(min(len(init_color_files), len(effect_color_files),
                                  len(init_depth_files), len(effect_depth_files))):
                    # Extract sample index from filename
                    try:
                        sample_idx = int(os.path.basename(init_color_files[i]).split('_')[-1].split('.')[0])
                        logger.info(f"      Processing sample {sample_idx}")

                        # Process this sample
                        process_sample(
                            object_idx,
                            object_name,
                            tool,
                            action,
                            init_color_files[i],
                            effect_color_files[i],
                            init_depth_files[i],
                            effect_depth_files[i]
                        )
                    except (IndexError, ValueError) as e:
                        logger.error(f"      Error processing sample {i}: {e}")
                        continue

                logger.info(f"    Completed action: {action}")

            logger.info(f"  Completed tool: {tool}")

        logger.info(f"Completed object {object_idx}: {object_name}")

    logger.info("Dataset processing complete!")


def main():
    """Main function to run the script."""
    logger.info("Starting image concatenation process")

    try:
        process_dataset()
        logger.info("Image concatenation completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)