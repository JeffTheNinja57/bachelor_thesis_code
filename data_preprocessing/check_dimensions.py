"""
check_dimensions.py - Check dimensions of output images

This script checks the dimensions of one image from each output directory
to verify that they match the requirements:
- Late fusion images should be 64x128 (two 64x64 images side by side)
- Early fusion images should be 64x256 (four 64x64 images side by side)
"""

from PIL import Image
import os

# Sample images to check
late_color_img = "data/late_color/0_hook_left_to_right_0_color.png"
late_depth_img = "data/late_depth/0_hook_left_to_right_0_depth.png"
early_img = "data/early/0_hook_left_to_right_0_early.png"

# Check dimensions
if os.path.exists(late_color_img):
    img = Image.open(late_color_img)
    print(f"Late color image dimensions: {img.size}")
    print(f"Expected: (128, 64)")
    print(f"Matches expected dimensions: {img.size == (128, 64)}")
    print()
else:
    print(f"Late color image not found: {late_color_img}")

if os.path.exists(late_depth_img):
    img = Image.open(late_depth_img)
    print(f"Late depth image dimensions: {img.size}")
    print(f"Expected: (128, 64)")
    print(f"Matches expected dimensions: {img.size == (128, 64)}")
    print()
else:
    print(f"Late depth image not found: {late_depth_img}")

if os.path.exists(early_img):
    img = Image.open(early_img)
    print(f"Early fusion image dimensions: {img.size}")
    print(f"Expected: (256, 64)")
    print(f"Matches expected dimensions: {img.size == (256, 64)}")
    print()
else:
    print(f"Early fusion image not found: {early_img}")