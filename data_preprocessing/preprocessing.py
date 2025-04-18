#!/usr/bin/env python3
"""
concat_images.py - Image Concatenation for Action Recognition Dataset

This script processes an action recognition dataset by concatenating images in two ways:
1. Late Fusion: Horizontally concatenate pairs of images (color-color and depth-depth)
2. Early Fusion: Horizontally concatenate all four images (color-color-depth-depth)

It traverses:
    datasets/action_recognition_dataset/
        <object_idx>_<objectName>/
            <tool>/
                <action>/
                    color/            <-- init_color_0.png ... init_color_9.png, effect_color_*.png
                    depthcolormap/    <-- init_depth_0.png ... init_depth_9.png, effect_depth_*.png

And writes out to:
    data/late_color/
    data/late_depth/
    data/early/
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

# --- CONSTANTS ---
TARGET_SIZE = (64, 64)  # each small image will be 64×64
DATASET_PATH = '../datasets/action_recognition_dataset'
OUTPUT_DIRS = {
    'late_color': 'data/late_color',
    'late_depth': 'data/late_depth',
    'early': 'data/early'
}
VALID_CAMERAS = ['color', 'depthcolormap']


def create_output_directories():
    """Ensure the three fusion output directories exist."""
    for d in OUTPUT_DIRS.values():
        os.makedirs(d, exist_ok=True)
        logger.debug(f"Ensured output dir exists: {d}")


def load_and_resize_image(image_path):
    """
    Load an image and resize to TARGET_SIZE if needed.
    Returns a PIL Image or None on failure.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        if img.size != TARGET_SIZE:
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
        return img
    except Exception as e:
        logger.error(f"Failed to load/resize '{image_path}': {e}")
        return None


def horizontal_concat_images(images):
    """
    Horizontally concatenate a list of PIL Images.
    Returns the concatenated image, or None if any element is None.
    """
    if any(img is None for img in images) or not images:
        return None

    total_width = sum(img.width for img in images)
    height = images[0].height
    canvas = Image.new('RGB', (total_width, height))

    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width

    return canvas


def process_sample(object_idx, tool, action, sample_idx,
                   init_color_path, effect_color_path,
                   init_depth_path, effect_depth_path):
    """
    Given the four file paths for one sample index, produce
    - late_color (64×128)
    - late_depth (64×128)
    - early (64×256)
    and save them.
    """
    # 1) Load & resize
    ic = load_and_resize_image(init_color_path)
    ec = load_and_resize_image(effect_color_path)
    idp = load_and_resize_image(init_depth_path)
    edp = load_and_resize_image(effect_depth_path)

    if None in (ic, ec, idp, edp):
        logger.error(f"  ← Skipping sample {sample_idx}, image load error")
        return

    # 2) Late fusion
    late_color = horizontal_concat_images([ic, ec])  # 64×128
    late_depth = horizontal_concat_images([idp, edp])  # 64×128

    # 3) Early fusion (all four side by side)
    early = horizontal_concat_images([ic, ec, idp, edp])  # 64×256

    # 4) Save with sample_idx in filename
    prefix = f"{object_idx}_{tool}_{action}_{sample_idx}"
    if late_color:
        late_color.save(os.path.join(OUTPUT_DIRS['late_color'], f"{prefix}_color.png"))
    if late_depth:
        late_depth.save(os.path.join(OUTPUT_DIRS['late_depth'], f"{prefix}_depth.png"))
    if early:
        early.save(os.path.join(OUTPUT_DIRS['early'], f"{prefix}_early.png"))

    logger.info(f"    Sample {sample_idx} done")


def build_index_map(file_list, expected_prefix):
    """
    From a list of file paths like /…/init_color_0.png, extract {0: path, 1: path, …}.
    Only keeps files whose basename starts with expected_prefix + '_<idx>.png'
    """
    d = {}
    for path in file_list:
        name = os.path.basename(path)
        parts = name.split('_')
        # last segment is like '5.png' → drop .png
        try:
            idx = int(parts[-1].split('.')[0])
        except ValueError:
            continue
        if name.startswith(expected_prefix + '_'):
            d[idx] = path
    return d


def process_dataset():
    create_output_directories()

    # iterate objects 0_… through 19_…
    for obj_dir in sorted(os.listdir(DATASET_PATH)):
        obj_path = os.path.join(DATASET_PATH, obj_dir)
        if not os.path.isdir(obj_path) or '_' not in obj_dir:
            continue

        idx_str, obj_name = obj_dir.split('_', 1)
        if not idx_str.isdigit():
            continue
        object_idx = int(idx_str)
        if not (0 <= object_idx <= 19):
            continue

        logger.info(f"Object {object_idx}: {obj_name}")

        # each tool
        for tool in sorted(os.listdir(obj_path)):
            tool_path = os.path.join(obj_path, tool)
            if not os.path.isdir(tool_path):
                continue
            logger.info(f"  Tool: {tool}")

            # each action
            for action in sorted(os.listdir(tool_path)):
                action_path = os.path.join(tool_path, action)
                if not os.path.isdir(action_path):
                    continue

                # ensure both views exist
                if not all(os.path.isdir(os.path.join(action_path, cam))
                           for cam in VALID_CAMERAS):
                    logger.warning(f"    Skipping {action}, missing camera dirs")
                    continue

                logger.info(f"    Action: {action}")

                # collect color files
                color_dir = os.path.join(action_path, 'color')
                init_color_files = glob.glob(os.path.join(color_dir, 'init_color_*.png'))
                effect_color_files = glob.glob(os.path.join(color_dir, 'effect_color_*.png'))

                # collect depth files
                depth_dir = os.path.join(action_path, 'depthcolormap')
                init_depth_files = glob.glob(os.path.join(depth_dir, 'init_depthcolormap_*.png'))
                effect_depth_files = glob.glob(os.path.join(depth_dir, 'effect_depthcolormap_*.png'))

                # build idx→path maps
                init_color_map = build_index_map(init_color_files, 'init_color')
                effect_color_map = build_index_map(effect_color_files, 'effect_color')
                init_depth_map = build_index_map(init_depth_files, 'init_depthcolormap')
                effect_depth_map = build_index_map(effect_depth_files, 'effect_depthcolormap')

                # find the common sample indices
                common_idxs = sorted(
                    set(init_color_map)
                    & set(effect_color_map)
                    & set(init_depth_map)
                    & set(effect_depth_map)
                )

                if not common_idxs:
                    logger.warning(f"      No matching samples in {action}")
                    continue

                # process each sample
                for sample_idx in common_idxs:
                    logger.debug(f"    → Sample {sample_idx}")
                    process_sample(
                        object_idx, tool, action, sample_idx,
                        init_color_map[sample_idx],
                        effect_color_map[sample_idx],
                        init_depth_map[sample_idx],
                        effect_depth_map[sample_idx]
                    )

    logger.info("All done.")


if __name__ == '__main__':
    logger.info("Starting concatenation...")
    process_dataset()
    logger.info("Finished.")
