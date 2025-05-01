import glob
import logging
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# Custom normalization transform for depth images
class DepthNormalize:
    """
    Custom normalization for depth images that handles the red channel differently.
    If the red channel has a standard deviation of 0, it is not normalized.
    """
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Check if the red channel (index 0) has a standard deviation > 0
        red_std = torch.std(tensor[0])

        # Create a copy of the tensor to avoid modifying the original
        normalized = tensor.clone()

        # Always normalize green and blue channels
        for i in range(1, 3):
            normalized[i] = (tensor[i] - self.mean[i]) / self.std[i]

        # Only normalize red channel if std > 0
        if red_std > 0:
            normalized[0] = (tensor[0] - self.mean[0]) / self.std[0]

        return normalized

# Get logger for this module
log = logging.getLogger(__name__)


class EarlyFusionDataset(Dataset):
    """
    Dataset class for early fusion that loads a single concatenated RGB image.

    Args:
        data_dir (str): Root directory where the dataset is stored.
        split (str): 'train', 'val', or 'test'.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.data_path = os.path.join(data_dir, 'early', split)

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")

        log.info(f"Loading early fusion data from: {self.data_path}")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            log.info("Using default ToTensor and Normalize transform.")
        else:
            self.transform = transform
            log.info("Using provided custom transform.")

        self.samples = []
        self.action_labels = []
        self.tool_labels = []
        self._load_dataset()

        # Set dataset properties
        self.num_action_classes = len(set(self.action_labels))
        self.num_tool_classes = len(set(self.tool_labels))

        # Set image dimensions based on the first image
        if len(self.samples) > 0:
            sample_img = Image.open(self.samples[0])
            self.img_height, self.img_width = sample_img.height, sample_img.width
            self.input_channels = 3  # RGB image
        else:
            log.warning("No samples found in the dataset!")
            self.img_height, self.img_width = 64, 256  # Default values from issue description
            self.input_channels = 3

        log.info(f"Dataset loaded with {len(self.samples)} samples")
        log.info(f"Action classes: {self.num_action_classes}, Tool classes: {self.num_tool_classes}")
        log.info(f"Image dimensions: {self.img_height}x{self.img_width}")

    def _load_dataset(self):
        """Load dataset files and extract class labels."""
        # For early fusion, load all images with _early.png suffix
        pattern = os.path.join(self.data_path, '*_early.png')
        files = glob.glob(pattern)

        # Define the action and tool classes
        actions = ["push", "pull", "left", "right"]
        tools = ["sshot", "spatula", "hook", "ruler"]

        action_to_idx = {action: idx for idx, action in enumerate(actions)}
        tool_to_idx = {tool: idx for idx, tool in enumerate(tools)}

        for file_path in files:
            # Extract class labels from filename
            # Format: <object_idx>_<tool>_<action>_<sample_idx>_early.png
            base_name = os.path.basename(file_path)
            parts = base_name.split('_')

            if len(parts) >= 4:  # Ensure we have enough parts
                tool = parts[1]
                action = parts[2]

                if tool in tool_to_idx and action in action_to_idx:
                    tool_label = tool_to_idx[tool]
                    action_label = action_to_idx[action]

                    self.samples.append(file_path)
                    self.action_labels.append(action_label)
                    self.tool_labels.append(tool_label)
                else:
                    log.warning(f"Unknown tool or action in filename: {base_name}")
            else:
                log.warning(f"Filename does not match expected format: {base_name}")

        if len(self.samples) == 0:
            log.warning(f"No samples found in {self.data_path}!")

        # Convert labels to numpy arrays for easier handling
        self.action_labels = np.array(self.action_labels)
        self.tool_labels = np.array(self.tool_labels)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Gets the image data and labels for a given index.

        Returns:
            tuple: (img, action_label, tool_label)
        """
        img_path = self.samples[idx]
        action_label = self.action_labels[idx]
        tool_label = self.tool_labels[idx]

        # Load early fusion image
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, action_label, tool_label

    def get_details(self):
        """
        Returns essential details about the dataset format.

        Returns:
            tuple: (input_channels, height, width, num_action_classes, num_tool_classes)
        """
        return self.input_channels, self.img_height, self.img_width, self.num_action_classes, self.num_tool_classes


class LateFusionColorDataset(Dataset):
    """
    Dataset class for late fusion that loads only color images.

    Args:
        data_dir (str): Root directory where the dataset is stored.
        split (str): 'train', 'val', or 'test'.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.color_path = os.path.join(data_dir, 'late_color', split)

        if not os.path.exists(self.color_path):
            raise FileNotFoundError(f"Data directory not found: {self.color_path}")

        log.info(f"Loading late fusion color data from: {self.color_path}")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            log.info("Using default ToTensor and Normalize transform.")
        else:
            self.transform = transform
            log.info("Using provided custom transform.")

        self.samples = []  # Will contain color image paths
        self.action_labels = []
        self.tool_labels = []
        self._load_dataset()

        # Set dataset properties
        self.num_action_classes = len(set(self.action_labels))
        self.num_tool_classes = len(set(self.tool_labels))

        # Set image dimensions based on the first image
        if len(self.samples) > 0:
            sample_img = Image.open(self.samples[0])
            self.img_height, self.img_width = sample_img.height, sample_img.width
            self.input_channels = 3  # RGB image
        else:
            log.warning("No samples found in the dataset!")
            self.img_height, self.img_width = 64, 128  # Default values
            self.input_channels = 3

        log.info(f"Color dataset loaded with {len(self.samples)} samples")
        log.info(f"Action classes: {self.num_action_classes}, Tool classes: {self.num_tool_classes}")
        log.info(f"Image dimensions: {self.img_height}x{self.img_width}")

    def _load_dataset(self):
        """Load dataset files and extract class labels."""
        # Load color images
        color_pattern = os.path.join(self.color_path, '*_color.png')
        color_files = glob.glob(color_pattern)

        # Define the action and tool classes
        actions = ["push", "pull", "left", "right"]
        tools = ["sshot", "spatula", "hook", "ruler"]

        action_to_idx = {action: idx for idx, action in enumerate(actions)}
        tool_to_idx = {tool: idx for idx, tool in enumerate(tools)}

        for color_file in color_files:
            # Extract class labels from filename
            base_name = os.path.basename(color_file)
            # Format: <object_idx>_<tool>_<action>_<sample_idx>_color.png
            parts = base_name.split('_')
            if len(parts) < 5:
                log.warning(f"Filename does not match expected format: {base_name}")
                continue

            tool = parts[1]
            action = parts[2]

            if tool in tool_to_idx and action in action_to_idx:
                tool_label = tool_to_idx[tool]
                action_label = action_to_idx[action]

                self.samples.append(color_file)
                self.action_labels.append(action_label)
                self.tool_labels.append(tool_label)
            else:
                log.warning(f"Unknown tool or action in filename: {base_name}")

        if len(self.samples) == 0:
            log.warning(f"No samples found in {self.color_path}!")

        # Convert labels to numpy arrays for easier handling
        self.action_labels = np.array(self.action_labels)
        self.tool_labels = np.array(self.tool_labels)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Gets the image data and labels for a given index.

        Returns:
            tuple: (img_color, action_label, tool_label)
        """
        color_path = self.samples[idx]
        action_label = self.action_labels[idx]
        tool_label = self.tool_labels[idx]

        # Load color image
        img_color = Image.open(color_path).convert('RGB')

        if self.transform:
            img_color = self.transform(img_color)

        return img_color, action_label, tool_label

    def get_details(self):
        """
        Returns essential details about the dataset format.

        Returns:
            tuple: (input_channels, height, width, num_action_classes, num_tool_classes)
        """
        return self.input_channels, self.img_height, self.img_width, self.num_action_classes, self.num_tool_classes


class LateFusionDepthDataset(Dataset):
    """
    Dataset class for late fusion that loads only depth images.

    Args:
        data_dir (str): Root directory where the dataset is stored.
        split (str): 'train', 'val', or 'test'.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.depth_path = os.path.join(data_dir, 'late_depth', split)

        if not os.path.exists(self.depth_path):
            raise FileNotFoundError(f"Data directory not found: {self.depth_path}")

        log.info(f"Loading late fusion depth data from: {self.depth_path}")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                DepthNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
            log.info("Using default ToTensor and DepthNormalize transform.")
        else:
            self.transform = transform
            log.info("Using provided custom transform.")

        self.samples = []  # Will contain depth image paths
        self.action_labels = []
        self.tool_labels = []
        self._load_dataset()

        # Set dataset properties
        self.num_action_classes = len(set(self.action_labels))
        self.num_tool_classes = len(set(self.tool_labels))

        # Set image dimensions based on the first image
        if len(self.samples) > 0:
            sample_img = Image.open(self.samples[0])
            self.img_height, self.img_width = sample_img.height, sample_img.width
            self.input_channels = 3  # RGB image
        else:
            log.warning("No samples found in the dataset!")
            self.img_height, self.img_width = 64, 128  # Default values
            self.input_channels = 3

        log.info(f"Depth dataset loaded with {len(self.samples)} samples")
        log.info(f"Action classes: {self.num_action_classes}, Tool classes: {self.num_tool_classes}")
        log.info(f"Image dimensions: {self.img_height}x{self.img_width}")

    def _load_dataset(self):
        """Load dataset files and extract class labels."""
        # Load depth images
        depth_pattern = os.path.join(self.depth_path, '*_depth.png')
        depth_files = glob.glob(depth_pattern)

        # Define the action and tool classes
        actions = ["push", "pull", "left", "right"]  # Match the action names used in color dataset
        tools = ["sshot", "spatula", "hook", "ruler"]

        action_to_idx = {action: idx for idx, action in enumerate(actions)}
        tool_to_idx = {tool: idx for idx, tool in enumerate(tools)}

        for depth_file in depth_files:
            # Extract class labels from filename
            base_name = os.path.basename(depth_file)
            # Format: <object_idx>_<tool>_<action>_<sample_idx>_depth.png
            parts = base_name.split('_')
            if len(parts) < 5:
                log.warning(f"Filename does not match expected format: {base_name}")
                continue

            tool = parts[1]
            action = parts[2]

            if tool in tool_to_idx and action in action_to_idx:
                tool_label = tool_to_idx[tool]
                action_label = action_to_idx[action]

                self.samples.append(depth_file)
                self.action_labels.append(action_label)
                self.tool_labels.append(tool_label)
            else:
                log.warning(f"Unknown tool or action in filename: {base_name}")

        if len(self.samples) == 0:
            log.warning(f"No samples found in {self.depth_path}!")

        # Convert labels to numpy arrays for easier handling
        self.action_labels = np.array(self.action_labels)
        self.tool_labels = np.array(self.tool_labels)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Gets the image data and labels for a given index.

        Returns:
            tuple: (img_depth, action_label, tool_label)
        """
        depth_path = self.samples[idx]
        action_label = self.action_labels[idx]
        tool_label = self.tool_labels[idx]

        # Load depth image
        img_depth = Image.open(depth_path).convert('RGB')

        if self.transform:
            img_depth = self.transform(img_depth)

        return img_depth, action_label, tool_label

    def get_details(self):
        """
        Returns essential details about the dataset format.

        Returns:
            tuple: (input_channels, height, width, num_action_classes, num_tool_classes)
        """
        return self.input_channels, self.img_height, self.img_width, self.num_action_classes, self.num_tool_classes


def get_dataloaders(config):
    """
    Build PyTorch DataLoaders for train/val/test from the appropriate data directories.

    Args:
        config (dict): Configuration dictionary containing:
            - fusion_type: "early" or "late"
            - data_dir: Root directory for the dataset
            - batch_size: Batch size for DataLoaders
            - num_workers: Number of workers for DataLoaders

    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset_info)
            For late fusion, returns (color_train_loader, color_val_loader, color_test_loader, 
                                     depth_train_loader, depth_val_loader, depth_test_loader, dataset_info)
            dataset_info is a dict containing:
                - input_channels: Number of input channels
                - img_height: Image height
                - img_width: Image width
                - num_action_classes: Number of action classes
                - num_tool_classes: Number of tool classes
    """
    fusion_type = config.get('fusion_type')
    data_dir = config.get('data_dir')
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 2)

    if fusion_type not in ['early', 'late']:
        raise ValueError(f"Invalid fusion_type: {fusion_type}. Must be 'early' or 'late'.")

    # Define transforms
    color_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    depth_transform = transforms.Compose([
        transforms.ToTensor(),
        DepthNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create datasets based on fusion type
    if fusion_type == 'early':
        train_dataset = EarlyFusionDataset(data_dir, split='train', transform=color_transform)
        val_dataset = EarlyFusionDataset(data_dir, split='val', transform=color_transform)
        test_dataset = EarlyFusionDataset(data_dir, split='test', transform=color_transform)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Get dataset details
        input_channels, img_height, img_width, num_action_classes, num_tool_classes = train_dataset.get_details()

        dataset_info = {
            'input_channels': input_channels,
            'img_height': img_height,
            'img_width': img_width,
            'num_action_classes': num_action_classes,
            'num_tool_classes': num_tool_classes
        }

        return train_loader, val_loader, test_loader, dataset_info

    else:  # late fusion
        # Create separate color and depth datasets
        color_train_dataset = LateFusionColorDataset(data_dir, split='train', transform=color_transform)
        color_val_dataset = LateFusionColorDataset(data_dir, split='val', transform=color_transform)
        color_test_dataset = LateFusionColorDataset(data_dir, split='test', transform=color_transform)

        depth_train_dataset = LateFusionDepthDataset(data_dir, split='train', transform=depth_transform)
        depth_val_dataset = LateFusionDepthDataset(data_dir, split='val', transform=depth_transform)
        depth_test_dataset = LateFusionDepthDataset(data_dir, split='test', transform=depth_transform)

        # For training, we need to ensure the same shuffling for both color and depth
        # Create a random seed for the shuffling
        seed = 42
        torch.manual_seed(seed)

        # Create a shared sampler for training to ensure same order
        train_indices = list(range(len(color_train_dataset)))
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

        # Create DataLoaders for color datasets
        color_train_loader = DataLoader(color_train_dataset, batch_size=batch_size, 
                                        sampler=train_sampler, num_workers=num_workers)
        color_val_loader = DataLoader(color_val_dataset, batch_size=batch_size, 
                                      shuffle=False, num_workers=num_workers)
        color_test_loader = DataLoader(color_test_dataset, batch_size=batch_size, 
                                       shuffle=False, num_workers=num_workers)

        # Reset seed to ensure same shuffling for depth
        torch.manual_seed(seed)

        # Create DataLoaders for depth datasets with the same shuffling as color
        depth_train_loader = DataLoader(depth_train_dataset, batch_size=batch_size, 
                                        sampler=train_sampler, num_workers=num_workers)
        depth_val_loader = DataLoader(depth_val_dataset, batch_size=batch_size, 
                                      shuffle=False, num_workers=num_workers)
        depth_test_loader = DataLoader(depth_test_dataset, batch_size=batch_size, 
                                       shuffle=False, num_workers=num_workers)

        # Get dataset details
        color_channels, img_height, img_width, num_action_classes, num_tool_classes = color_train_dataset.get_details()
        depth_channels = depth_train_dataset.input_channels

        dataset_info = {
            'input_channels': (color_channels, depth_channels),  # Tuple of (color_channels, depth_channels)
            'img_height': img_height,
            'img_width': img_width,
            'num_action_classes': num_action_classes,
            'num_tool_classes': num_tool_classes
        }

        # For late fusion, return both color and depth loaders
        return (color_train_loader, color_val_loader, color_test_loader,
                depth_train_loader, depth_val_loader, depth_test_loader,
                dataset_info)
