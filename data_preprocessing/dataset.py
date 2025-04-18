import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import logging

# Configure basic logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)


class ActionDataset(Dataset):
    """

    Dataset class for loading action recognition data.

    *** IMPORTANT NOTE ***
    This implementation SIMULATES the required 4-image input
    (start_color, end_color, start_depth, end_depth) using the MNIST dataset
    as requested by the user. Each MNIST digit image is replicated four times.
    This is ONLY for demonstrating the code structure and WILL NOT perform
    meaningful action recognition or fusion comparison.

    Replace this with your actual dataset loading logic for your specific
    action recognition image files and directory structure.
    ********************

    Args:
        data_dir (str): Directory where MNIST data is/will be stored (e.g., './data').
        split (str): 'train' or 'test'.
        fusion_type (str): 'early' or 'late'. Determines output format.
        transform (callable, optional): Optional transform to be applied on a sample.
                                        Defaults to basic ToTensor and Normalize.
        download (bool): If true, downloads the dataset from the internet if not present.
    """

    def __init__(self, data_dir, split='train', fusion_type='early', transform=None, download=True):
        log.warning("Initializing ActionDataset using SIMULATED data from MNIST.")
        self.data_dir = data_dir
        self.split = split
        self.fusion_type = fusion_type
        self.is_train = (split == 'train')

        if fusion_type not in ['early', 'late']:
            raise ValueError("fusion_type must be 'early' or 'late'")

        # Define default transformations for MNIST (1 channel, 28x28)
        if transform is None:
            # MNIST mean and std are approx 0.1307, 0.3081
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            log.info("Using default ToTensor and Normalize transform for MNIST.")
        else:
            self.transform = transform
            log.info("Using provided custom transform.")

        # --- Load MNIST Data ---
        mnist_data_path = os.path.join(data_dir, 'MNIST')
        log.info(f"Loading MNIST data from: {mnist_data_path} (Train={self.is_train}, Download={download})")
        try:
            self.mnist_dataset = torchvision.datasets.MNIST(
                root=mnist_data_path,
                train=self.is_train,
                download=download,
                transform=self.transform  # Apply transform here
            )
        except Exception as e:
            log.error(f"Failed to load or download MNIST dataset from {mnist_data_path}: {e}")
            raise

        # --- Define simulated properties ---
        self.num_classes = 10  # MNIST has 10 classes (digits 0-9)
        self.img_height = 28
        self.img_width = 28

        # Define effective input channels based on simulation and fusion type
        if self.fusion_type == 'early':
            # 4 simulated images concatenated (1 channel each)
            self.input_channels = 4
            log.info(f"Early fusion mode: Outputting single tensor with {self.input_channels} channels.")
        else:  # Late fusion
            # 2 simulated color images (1 channel each -> 2 channels)
            # 2 simulated depth images (1 channel each -> 2 channels)
            self.input_channels = (2, 2)  # Tuple for color_channels, depth_channels
            log.info(f"Late fusion mode: Outputting two tensors with channels {self.input_channels}.")

    def __len__(self):
        """Return the total number of samples."""
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        """
        Gets the simulated 4-image data and label for a given index.

        Returns:
            if fusion_type == 'early':
                tuple: (tensor, label) where tensor is [4, H, W]
            if fusion_type == 'late':
                tuple: (color_tensor, depth_tensor, label) where tensors are [2, H, W]
        """
        # Get the single MNIST image and its label
        mnist_img_tensor, label = self.mnist_dataset[idx]
        # mnist_img_tensor shape is [1, H, W] after ToTensor transform

        # Simulate the four images by replicating the tensor
        # In a real dataset, you would load four different images here.
        img_start_color = mnist_img_tensor  # Shape: [1, H, W]
        img_end_color = mnist_img_tensor  # Shape: [1, H, W]
        img_start_depth = mnist_img_tensor  # Shape: [1, H, W] (using grayscale as depth)
        img_end_depth = mnist_img_tensor  # Shape: [1, H, W]

        # --- Format output based on fusion type ---
        if self.fusion_type == 'early':
            # Concatenate along the channel dimension (dim=0)
            fused_tensor = torch.cat(
                (img_start_color, img_end_color, img_start_depth, img_end_depth),
                dim=0
            )  # Shape: [4, H, W]
            return fused_tensor, label

        else:  # Late fusion
            # Concatenate color images
            color_tensor = torch.cat((img_start_color, img_end_color), dim=0)  # Shape: [2, H, W]
            # Concatenate depth images
            depth_tensor = torch.cat((img_start_depth, img_end_depth), dim=0)  # Shape: [2, H, W]
            return color_tensor, depth_tensor, label

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