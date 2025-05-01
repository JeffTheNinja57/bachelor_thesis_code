import os
import random
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Source directories
EARLY_DIR = '../data/early'
LATE_COLOR_DIR = '../data/late_color'
LATE_DEPTH_DIR = '../data/late_depth'

# How to split
TRAIN_RATIO = 0.7
TEST_RATIO = 0.15
VAL_RATIO = 0.15  # must sum to 1.0

# Where to deposit the splits
OUTPUT_DIRS = {
    'early': {
        'train': 'data/early/train',
        'test': 'data/early/test',
        'val': 'data/early/val',
    },
    'late_color': {
        'train': 'data/late_color/train',
        'test': 'data/late_color/test',
        'val': 'data/late_color/val',
    },
    'late_depth': {
        'train': 'data/late_depth/train',
        'test': 'data/late_depth/test',
        'val': 'data/late_depth/val',
    },
}

# Exact suffix each fusion type uses on disk
SUFFIXES = {
    'early': '_early.png',
    'late_color': '_color.png',
    'late_depth': '_depth.png',
}


def create_output_directories():
    """Make sure all split dirs exist."""
    for fusion_type, splits in OUTPUT_DIRS.items():
        for split_name, path in splits.items():
            os.makedirs(path, exist_ok=True)
            logger.debug(f"Ensured directory: {path}")


def get_base_filenames_from_early():
    """
    Look in EARLY_DIR, find all files ending in '_early.png',
    and strip off that suffix to yield the 'base name' for each sample.
    E.g. '0_hook_left_to_right_3_early.png' → '0_hook_left_to_right_3'
    """
    files = os.listdir(EARLY_DIR)
    base_names = []
    suffix = SUFFIXES['early']
    for fname in files:
        if not fname.endswith(suffix):
            continue
        base = fname[:-len(suffix)]
        base_names.append(base)
    logger.info(f"Found {len(base_names)} samples in {EARLY_DIR}")
    return base_names


def split_dataset():
    create_output_directories()

    # 1) collect and shuffle
    all_bases = get_base_filenames_from_early()
    random.shuffle(all_bases)

    # 2) compute splits
    N = len(all_bases)
    n_train = int(N * TRAIN_RATIO)
    n_test = int(N * TEST_RATIO)
    # rest goes to val
    n_val = N - n_train - n_test

    train_bases = all_bases[:n_train]
    test_bases = all_bases[n_train:n_train + n_test]
    val_bases = all_bases[n_train + n_test:]

    logger.info(f"Split sizes → train: {len(train_bases)}, test: {len(test_bases)}, val: {len(val_bases)}")

    # 3) copy for each fusion_type
    for fusion_type in ['early', 'late_color', 'late_depth']:
        for split_name, bases in [('train', train_bases),
                                  ('test', test_bases),
                                  ('val', val_bases)]:
            copy_split(fusion_type, split_name, bases)


def copy_split(fusion_type, split_name, base_names):
    """
    For a given fusion_type and split (train/test/val), copy each
    <base><suffix> from the source dir into the corresponding OUTPUT_DIR.
    """
    src_dir = {
        'early': EARLY_DIR,
        'late_color': LATE_COLOR_DIR,
        'late_depth': LATE_DEPTH_DIR
    }[fusion_type]

    dst_dir = OUTPUT_DIRS[fusion_type][split_name]
    suffix = SUFFIXES[fusion_type]

    count = 0
    for base in base_names:
        src_path = os.path.join(src_dir, base + suffix)
        dst_path = os.path.join(dst_dir, base + suffix)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            count += 1
        else:
            logger.warning(f"Missing {fusion_type} file for '{base}': expected {src_path}")

    logger.info(f"[{fusion_type}][{split_name}] copied {count}/{len(base_names)} files → {dst_dir}")


def main():
    logger.info("Starting dataset split …")
    split_dataset()
    logger.info("Done.")


if __name__ == "__main__":
    main()
