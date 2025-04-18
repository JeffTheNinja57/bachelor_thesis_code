import torch
import torch.nn as nn
import logging


# --- Model Building Function ---

def build_cnn(architecture_encoding, input_channels, num_classes, config):
    """
    Builds a PyTorch CNN model dynamically based on the architecture encoding.

    Args:
        architecture_encoding (list): List of layer definition dictionaries.
                                      Example: [{'type': 'conv', 'out_channels': 32, ...}, ...]
        input_channels (int): Number of channels in the input image.
        num_classes (int): Number of output classes for the final layer.
        config (dict): Configuration dictionary (potentially used for details like activation).

    Returns:
        nn.Module: The constructed PyTorch model.

    Raises:
        ValueError: If the architecture encoding is invalid or leads to build errors.
    """
    logging.info(f"Building CNN with {len(architecture_encoding)} layers...")
    layers = nn.ModuleList()
    current_channels = input_channels
    fc_input_features = None  # Track features entering the first FC layer
    is_conv_part = True  # Flag to track if we are still in Conv/Pool section

    # --- Layer Creation Loop ---
    for i, layer_def in enumerate(architecture_encoding):
        layer_type = layer_def.get('type')
        is_last_layer = (i == len(architecture_encoding) - 1)

        try:
            if layer_type == 'conv':
                if not is_conv_part:
                    raise ValueError("Conv layer found after FC layer definition.")

                kernel_size = layer_def['kernel_size']
                out_channels = layer_def['out_channels']
                stride = layer_def.get('stride', 1)
                # Calculate padding for 'same' effect if stride is 1
                padding = (kernel_size - 1) // 2 if stride == 1 else 0

                conv_layer = nn.Conv2d(current_channels, out_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding)
                layers.append(conv_layer)
                current_channels = out_channels
                # Add activation (ReLU) after Conv, unless followed immediately by BN
                if not (i + 1 < len(architecture_encoding) and architecture_encoding[i + 1].get('type') == 'bn'):
                    layers.append(nn.ReLU(inplace=True))

            elif layer_type == 'pool':
                if not is_conv_part:
                    raise ValueError("Pool layer found after FC layer definition.")

                pool_type = layer_def.get('pool_type', 'max')
                kernel_size = layer_def['kernel_size']
                stride = layer_def.get('stride', 2)  # Default stride 2 for pooling
                padding = layer_def.get('padding', 0)

                if pool_type == 'max':
                    pool_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
                elif pool_type == 'avg':
                    pool_layer = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
                else:
                    raise ValueError(f"Unsupported pool type: {pool_type}")
                layers.append(pool_layer)
                # Channel number doesn't change after pooling

            elif layer_type == 'bn':
                if is_conv_part:
                    # BatchNorm2d after Conv/Pool
                    bn_layer = nn.BatchNorm2d(current_channels)
                else:
                    # BatchNorm1d after FC
                    # Requires knowing the number of features from the preceding Linear layer
                    if fc_input_features is None:  # Should have been set by the Linear layer
                        raise ValueError("BatchNorm1d added without preceding Linear layer in FC part.")
                    bn_layer = nn.BatchNorm1d(fc_input_features)  # Apply BN on features from previous Linear

                layers.append(bn_layer)
                # Add activation (ReLU) after BN
                layers.append(nn.ReLU(inplace=True))


            elif layer_type == 'fc':
                neurons = layer_def['neurons']

                if is_conv_part:
                    # Transition from Conv/Pool to FC part
                    is_conv_part = False
                    # Use Adaptive Average Pooling to flatten spatial dimensions
                    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                    layers.append(nn.Flatten())
                    fc_input_features = current_channels  # Features are channels after AdaptiveAvgPool
                elif fc_input_features is None:
                    # This case should ideally not happen if arch starts with Conv
                    raise ValueError("FC layer encountered without preceding Conv/Pool or FC layer.")

                fc_layer = nn.Linear(fc_input_features, neurons)
                layers.append(fc_layer)
                fc_input_features = neurons  # Output features for the next layer

                # Add activation (ReLU) after FC, unless it's the final output layer
                # or followed immediately by BN
                if not is_last_layer and not \
                        (i + 1 < len(architecture_encoding) and architecture_encoding[i + 1].get('type') == 'bn'):
                    layers.append(nn.ReLU(inplace=True))

            elif layer_type == 'dropout':
                rate = layer_def.get('rate', 0.5)
                # Use Dropout (1D) as it typically follows FC/Activations
                layers.append(nn.Dropout(p=rate))

            else:
                raise ValueError(f"Unsupported layer type in architecture: {layer_type}")

        except KeyError as e:
            raise ValueError(f"Missing required parameter '{e}' for layer type '{layer_type}' at index {i}")
        except Exception as e:
            raise ValueError(f"Error building layer {i + 1} ({layer_type}): {e}")

    # --- Define the Model Class ---
    class DynamicCNN(nn.Module):
        def __init__(self, module_list):
            super().__init__()
            self.features = module_list
            self._initialize_weights()

        def forward(self, x):
            return self.features(x)  # Works if layers is nn.Sequential
            # If using ModuleList:
            # for layer in self.features:
            #     x = layer(x)
            # return x

        def _initialize_weights(self):
            logging.debug("Applying Xavier Uniform initialization...")
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    logging.debug(f"Initialized Conv2d: {m}")
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    logging.debug(f"Initialized Linear: {m}")
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.constant_(m.weight, 1)  # Standard BN init
                    nn.init.constant_(m.bias, 0)
                    logging.debug(f"Initialized BatchNorm: {m}")

    # --- Instantiate and Return ---
    try:
        # Using Sequential is simpler if the layer flow is linear
        model = nn.Sequential(*layers)
        # model = DynamicCNN(layers) # Use this if forward pass needs custom logic
        logging.info(f"CNN built successfully with {len(layers)} total modules.")
        # Log model structure and parameter count
        logging.debug(f"Model Structure:\n{model}")
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model Parameters (trainable): {params:,}")

    except Exception as e:
        logging.error(f"Error during final model instantiation: {e}")
        logging.error(f"Generated layers list: {layers}")
        raise ValueError(f"Failed to instantiate Sequential model: {e}")

    return model


# --- Example Usage (Standalone Test) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    print("--- Testing models/cnn_architecture.py ---")

    # Example Architecture Encoding (mimics one from pso.py test)
    test_arch = [
        {'type': 'conv', 'out_channels': 8, 'kernel_size': 3, 'stride': 1},  # L1
        {'type': 'bn'},  # L2
        # ReLU added automatically after BN
        {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2},  # L3 (No ReLU after pool)
        {'type': 'conv', 'out_channels': 16, 'kernel_size': 3, 'stride': 1},  # L4
        # ReLU added automatically (no BN follows)
        {'type': 'bn'},  # L5
        # ReLU added automatically after BN
        # AdaptiveAvgPool + Flatten added automatically before first FC
        {'type': 'fc', 'neurons': 32},  # L6
        {'type': 'bn'},  # L7
        # ReLU added automatically after BN
        {'type': 'dropout', 'rate': 0.25},  # L8
        {'type': 'fc', 'neurons': 5}  # L9 (Final layer, no ReLU)
    ]

    test_config = {}  # Empty config for this test
    test_input_channels = 3
    test_num_classes = 5
    # Assume dummy input size for testing calculation trace (if not using AdaptiveAvgPool)
    test_input_height = 32
    test_input_width = 32

    print("\nBuilding model with test architecture...")
    try:
        model = build_cnn(test_arch, test_input_channels, test_num_classes, test_config)
        print("\nModel built successfully!")
        # print(model) # Print structure

        # Test forward pass with dummy data
        print("\nTesting forward pass...")
        # Create dummy input tensor
        # Batch size = 4, Channels = 3, Height = 32, Width = 32
        dummy_input = torch.randn(4, test_input_channels, test_input_height, test_input_width)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        # Check if output shape matches (Batch size, num_classes)
        assert output.shape == (4, test_num_classes)
        print("Forward pass successful!")

    except ValueError as e:
        print(f"\nError building model: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    print("\n--- Test Complete ---")
