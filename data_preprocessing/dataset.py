import glob
import logging
import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

# Configure basic logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)


class ActionDataset(Dataset):
    """
    Dataset class for loading action recognition data.

    This implementation loads the actual action recognition data from the specified directories.
    It supports both early and late fusion modes.

    Args:
        data_dir (str): Root directory where the dataset is stored.
        split (str): 'train', 'test', or 'val'.
        fusion_type (str): 'early' or 'late'. Determines output format.
        transform (callable, optional): Optional transform to be applied on a sample.
                                        Defaults to basic ToTensor and Normalize.
        download (bool): Not used, kept for compatibility.
    """

    def __init__(self, data_dir, split='train', fusion_type='early', transform=None, download=True):
        self.data_dir = data_dir
        self.split = split
        self.fusion_type = fusion_type

        if fusion_type not in ['early', 'late']:
            raise ValueError("fusion_type must be 'early' or 'late'")

        # Define paths based on fusion type
        if fusion_type == 'early':
            self.data_path = os.path.join(data_dir, 'early', split)
            log.info(f"Loading early fusion data from: {self.data_path}")
        else:  # late fusion
            self.color_path = os.path.join(data_dir, 'late_color', split)
            self.depth_path = os.path.join(data_dir, 'late_depth', split)
            log.info(f"Loading late fusion data from: {self.color_path} and {self.depth_path}")

        # Define default transformations if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # Generic normalization, adjust as needed
            ])
            log.info("Using default ToTensor and Normalize transform.")
        else:
            self.transform = transform
            log.info("Using provided custom transform.")

        # Load file paths and labels
        self.samples = []
        self.labels = []
        self._load_dataset()

        # Set dataset properties
        self.num_classes = len(set(self.labels))

        # Set image dimensions based on the first image
        if len(self.samples) > 0:
            if fusion_type == 'early':
                sample_img = Image.open(self.samples[0])
                self.img_height, self.img_width = sample_img.height, sample_img.width
                self.input_channels = 4  # Assuming 4 channels for early fusion
            else:  # late fusion
                color_img = Image.open(self.samples[0][0])
                depth_img = Image.open(self.samples[0][1])
                self.img_height, self.img_width = color_img.height, color_img.width
                self.input_channels = (2, 2)  # Tuple for color_channels, depth_channels
        else:
            log.warning("No samples found in the dataset!")
            self.img_height, self.img_width = 224, 224  # Default values
            self.input_channels = 4 if fusion_type == 'early' else (2, 2)

        log.info(f"Dataset loaded with {len(self.samples)} samples, {self.num_classes} classes")
        log.info(f"Image dimensions: {self.img_height}x{self.img_width}")
        if fusion_type == 'early':
            log.info(f"Early fusion mode: Outputting single tensor with {self.input_channels} channels.")
        else:
            log.info(f"Late fusion mode: Outputting two tensors with channels {self.input_channels}.")

    def _load_dataset(self):
        """Load dataset files and extract class labels."""
        if self.fusion_type == 'early':
            # For early fusion, load all images with _early.png suffix
            pattern = os.path.join(self.data_path, '*_early.png')
            files = glob.glob(pattern)

            for file_path in files:
                # Extract class label from filename
                # Assuming format like: class_name_other_info_early.png
                base_name = os.path.basename(file_path)
                class_name = base_name.split('_')[0]  # Adjust based on your naming convention

                try:
                    label = int(class_name)
                except ValueError:
                    # If class_name is not an integer, create a mapping
                    # This is a simplified approach; you might need a more robust solution
                    if not hasattr(self, 'class_to_idx'):
                        self.class_to_idx = {}

                    if class_name not in self.class_to_idx:
                        self.class_to_idx[class_name] = len(self.class_to_idx)

                    label = self.class_to_idx[class_name]

                self.samples.append(file_path)
                self.labels.append(label)
        else:  # late fusion
            # For late fusion, load corresponding color and depth images
            color_pattern = os.path.join(self.color_path, '*_color.png')
            color_files = glob.glob(color_pattern)

            for color_file in color_files:
                # Find corresponding depth file
                base_name = os.path.basename(color_file)
                base_name = base_name.replace('_color.png', '')
                depth_file = os.path.join(self.depth_path, f"{base_name}_depth.png")

                if os.path.exists(depth_file):
                    # Extract class label from filename
                    class_name = base_name.split('_')[0]  # Adjust based on your naming convention

                    try:
                        label = int(class_name)
                    except ValueError:
                        # If class_name is not an integer, create a mapping
                        if not hasattr(self, 'class_to_idx'):
                            self.class_to_idx = {}

                        if class_name not in self.class_to_idx:
                            self.class_to_idx[class_name] = len(self.class_to_idx)

                        label = self.class_to_idx[class_name]

                    self.samples.append((color_file, depth_file))
                    self.labels.append(label)
                else:
                    log.warning(f"Missing depth file for {base_name}")

        if len(self.samples) == 0:
            log.warning(f"No samples found for {self.fusion_type} fusion in {self.split} split!")

        # Convert labels to numpy array for easier handling
        self.labels = np.array(self.labels)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Gets the actual image data and label for a given index.

        Returns:
            if fusion_type == 'early':
                tuple: (tensor, label) where tensor is [4, H, W]
            if fusion_type == 'late':
                tuple: (color_tensor, depth_tensor, label) where tensors are [2, H, W]
        """
        label = self.labels[idx]

        if self.fusion_type == 'early':
            # Load early fusion image
            img_path = self.samples[idx]
            img = Image.open(img_path).convert('RGB')  # Ensure it's RGB

            if self.transform:
                img = self.transform(img)

            # For early fusion, we assume the image already has all information
            # If your early fusion images are not already combined, you'll need to modify this
            return img, label

        else:  # Late fusion
            # Load color and depth images
            color_path, depth_path = self.samples[idx]

            color_img = Image.open(color_path).convert('RGB')
            depth_img = Image.open(depth_path).convert('RGB')

            if self.transform:
                color_img = self.transform(color_img)
                depth_img = self.transform(depth_img)

            # For late fusion, return separate color and depth tensors
            return color_img, depth_img, label

    def get_details(self):
        """
        Returns essential details about the dataset format.

        Returns:
            tuple: (input_channels, height, width, num_classes)
                   input_channels is int for early fusion, tuple for late fusion.
        """
        return self.input_channels, self.img_height, self.img_width, self.num_classes


# --- Example Usage (Standalone Test) ---
if __name__ == '__main__':
    log.setLevel(logging.DEBUG)  # Show more detail for testing
    print("--- Testing data/dataset.py (Simulated with MNIST) ---")

    # Create dummy data directory
    test_data_dir = "./test_mnist_data"
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    print("\nTesting Early Fusion Dataset:")
    try:
        early_dataset = ActionDataset(data_dir=test_data_dir, split='train', fusion_type='early', download=True)
        channels, h, w, classes = early_dataset.get_details()
        print(f" Details: Channels={channels}, H={h}, W={w}, Classes={classes}")
        print(f" Dataset length: {len(early_dataset)}")
        # Get one sample
        sample_data, sample_label = early_dataset[0]
        print(f" Sample 0: Data shape={sample_data.shape}, Label={sample_label}")
        assert sample_data.shape == (4, 28, 28)
        assert isinstance(sample_label, int)
        print(" Early fusion test PASSED.")
    except Exception as e:
        print(f" Early fusion test FAILED: {e}")

    print("\nTesting Late Fusion Dataset:")
    try:
        late_dataset = ActionDataset(data_dir=test_data_dir, split='test', fusion_type='late', download=True)
        channels, h, w, classes = late_dataset.get_details()
        print(f" Details: Channels={channels}, H={h}, W={w}, Classes={classes}")
        print(f" Dataset length: {len(late_dataset)}")
        # Get one sample
        color_data, depth_data, sample_label = late_dataset[0]
        print(f" Sample 0: Color shape={color_data.shape}, Depth shape={depth_data.shape}, Label={sample_label}")
        assert color_data.shape == (2, 28, 28)
        assert depth_data.shape == (2, 28, 28)
        assert isinstance(sample_label, int)
        print(" Late fusion test PASSED.")
    except Exception as e:
        print(f" Late fusion test FAILED: {e}")

    print("\n--- Test Complete ---")
    # Consider removing test_mnist_data directory after testing if desired
    # import shutil
    # shutil.rmtree(test_data_dir)
