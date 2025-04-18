import torch
import torch.nn as nn
import logging
from .cnn_architecture import build_cnn

# --- Early Fusion ---

def build_early_fusion_model(architecture_encoding, input_channels, num_classes, config):
    """
    Builds an Early Fusion model where inputs are fused before entering a single CNN.

    Args:
        architecture_encoding (list): Architecture encoding for the single CNN.
        input_channels (int): Total number of channels AFTER fusing inputs (e.g., 4 for simulated MNIST).
        num_classes (int): Number of output classes.
        config (dict): Configuration dictionary.

    Returns:
        nn.Module: The constructed Early Fusion PyTorch model.
    """
    logging.info(f"Building Early Fusion model with {input_channels} input channels.")
    # Early fusion simply uses the standard CNN builder with the combined input channels
    model = build_cnn(
        architecture_encoding=architecture_encoding,
        input_channels=input_channels,
        num_classes=num_classes,
        config=config
    )
    return model


# --- Late Fusion ---

class LateFusionModel(nn.Module):
    """
    Custom nn.Module for Late Fusion.
    Processes color and depth streams through separate CNN branches (sharing architecture)
    and fuses their features before final classification layers.
    """

    def __init__(self, shared_architecture_encoding, input_channels_color, input_channels_depth, num_classes, config):
        super().__init__()
        logging.info("Initializing Late Fusion Model...")
        self.config = config

        # --- Build Branches ---
        # Build the full sequential models first using the shared architecture
        logging.info("Building Color Branch...")
        full_color_branch = build_cnn(shared_architecture_encoding, input_channels_color, num_classes, config)
        logging.info("Building Depth Branch...")
        full_depth_branch = build_cnn(shared_architecture_encoding, input_channels_depth, num_classes, config)

        # --- Extract Feature Extractors ---
        # Find the index of the first Linear layer (after Flatten) to separate features/classifier
        first_linear_idx_color = -1
        first_linear_in_features_color = -1
        for i, layer in enumerate(full_color_branch):
            if isinstance(layer, nn.Linear):
                first_linear_idx_color = i
                first_linear_in_features_color = layer.in_features
                break
        if first_linear_idx_color == -1:
            raise ValueError("Could not find a Linear layer in the built color branch for feature extraction.")

        first_linear_idx_depth = -1
        first_linear_in_features_depth = -1
        for i, layer in enumerate(full_depth_branch):
            if isinstance(layer, nn.Linear):
                first_linear_idx_depth = i
                first_linear_in_features_depth = layer.in_features
                break
        if first_linear_idx_depth == -1:
            raise ValueError("Could not find a Linear layer in the built depth branch for feature extraction.")

        logging.info(f"Color branch feature size: {first_linear_in_features_color}")
        logging.info(f"Depth branch feature size: {first_linear_in_features_depth}")

        # Create feature extractors by taking layers up to the first Linear layer
        self.color_feature_extractor = nn.Sequential(*list(full_color_branch.children())[:first_linear_idx_color])
        self.depth_feature_extractor = nn.Sequential(*list(full_depth_branch.children())[:first_linear_idx_depth])

        # --- Define Shared Classifier ---
        # Input features to the classifier = combined features from both branches
        combined_feature_size = first_linear_in_features_color + first_linear_in_features_depth
        logging.info(f"Combined feature size for fusion: {combined_feature_size}")

        # Option 1: Use the classifier part from one of the built branches (if structure matches)
        # Assumes the FC layers defined in the architecture are the shared classifier
        # self.shared_classifier = nn.Sequential(*list(full_color_branch.children())[first_linear_idx_color:])

        # Option 2: Define a new shared classifier (more flexible)
        # Example: A single Linear layer for classification
        self.shared_classifier = nn.Sequential(
            # Optional: Add intermediate FC layers here if desired
            # nn.Linear(combined_feature_size, intermediate_features),
            # nn.ReLU(),
            # nn.Dropout(p=config.get('dropout_rate', 0.5) if config.get('use_dropout') else 0.0),
            nn.Linear(combined_feature_size, num_classes)  # Final classification layer
        )
        logging.info(f"Shared classifier structure:\n{self.shared_classifier}")
        self._initialize_classifier_weights()

    def _initialize_classifier_weights(self):
        # Initialize weights specifically for the newly defined classifier part
        logging.debug("Initializing Late Fusion shared classifier weights...")
        for m in self.shared_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                logging.debug(f"Initialized Shared Linear: {m}")
            # Add BN init if BN layers are included in the shared classifier
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                logging.debug(f"Initialized Shared BatchNorm1d: {m}")

    def forward(self, x_color, x_depth):
        """Forward pass for the Late Fusion model."""
        # Extract features from each branch
        features_color = self.color_feature_extractor(x_color)
        features_depth = self.depth_feature_extractor(x_depth)

        # Concatenate features
        # Features should be flattened already by the feature extractors (due to Flatten layer)
        fused_features = torch.cat((features_color, features_depth), dim=1)

        # Pass through shared classifier
        output = self.shared_classifier(fused_features)
        return output


def build_late_fusion_model(shared_architecture_encoding, input_channels_color, input_channels_depth, num_classes,
                            config):
    """
    Builds a Late Fusion model using a shared architecture for color and depth branches.

    Args:
        shared_architecture_encoding (list): Architecture encoding used for BOTH branches.
        input_channels_color (int): Number of input channels for the color stream.
        input_channels_depth (int): Number of input channels for the depth stream.
        num_classes (int): Number of output classes.
        config (dict): Configuration dictionary.

    Returns:
        LateFusionModel: The constructed Late Fusion PyTorch model.
    """
    logging.info("Building Late Fusion model (shared architecture)...")
    model = LateFusionModel(
        shared_architecture_encoding=shared_architecture_encoding,
        input_channels_color=input_channels_color,
        input_channels_depth=input_channels_depth,
        num_classes=num_classes,
        config=config
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Late Fusion Model Parameters (trainable): {params:,}")
    return model


# --- Example Usage (Standalone Test) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    print("--- Testing models/fusion_models.py ---")

    # Example shared architecture (simple)
    test_arch = [
        {'type': 'conv', 'out_channels': 8, 'kernel_size': 3, 'stride': 1},
        {'type': 'bn'},
        # ReLU added by build_cnn
        {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2},
        # AdaptiveAvgPool + Flatten added by build_cnn before FC
        {'type': 'fc', 'neurons': 16},  # Intermediate FC layer
        {'type': 'bn'},
        # ReLU added by build_cnn
        {'type': 'fc', 'neurons': 5}  # Final classification layer (used by build_cnn, replaced in LateFusionModel)
    ]
    test_config = {'use_bn': True}  # Example config detail
    test_num_classes = 5

    # --- Test Early Fusion ---
    print("\nTesting Early Fusion Model Build:")
    test_input_channels_early = 4  # e.g., 4 simulated MNIST channels
    try:
        early_model = build_early_fusion_model(test_arch, test_input_channels_early, test_num_classes, test_config)
        print("Early Fusion model built successfully.")
        # print(early_model)
        # Test forward pass
        dummy_input_early = torch.randn(2, test_input_channels_early, 28, 28)  # Batch=2
        with torch.no_grad():
            output_early = early_model(dummy_input_early)
        print(f"Early Fusion Input: {dummy_input_early.shape}, Output: {output_early.shape}")
        assert output_early.shape == (2, test_num_classes)
        print("Early Fusion forward pass successful.")
    except Exception as e:
        print(f"Error building/testing Early Fusion model: {e}")
        import traceback

        traceback.print_exc()

    # --- Test Late Fusion ---
    print("\nTesting Late Fusion Model Build:")
    test_input_channels_color = 2  # e.g., 2 simulated color channels
    test_input_channels_depth = 2  # e.g., 2 simulated depth channels
    try:
        late_model = build_late_fusion_model(test_arch, test_input_channels_color, test_input_channels_depth,
                                             test_num_classes, test_config)
        print("Late Fusion model built successfully.")
        # print(late_model)
        # Test forward pass
        dummy_input_color = torch.randn(3, test_input_channels_color, 28, 28)  # Batch=3
        dummy_input_depth = torch.randn(3, test_input_channels_depth, 28, 28)
        with torch.no_grad():
            output_late = late_model(dummy_input_color, dummy_input_depth)
        print(f"Late Fusion Inputs: Color={dummy_input_color.shape}, Depth={dummy_input_depth.shape}")
        print(f"Late Fusion Output: {output_late.shape}")
        assert output_late.shape == (3, test_num_classes)
        print("Late Fusion forward pass successful.")
    except Exception as e:
        print(f"Error building/testing Late Fusion model: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Test Complete ---")
