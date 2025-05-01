# project_root/utils/train.py

import math
import time

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm  # For progress bars


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        dataloader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): The device to train on ('mps' or 'cpu').

    Returns:
        tuple: Average epoch loss (float), average epoch accuracy (float).
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Wrap dataloader with tqdm for a progress bar
    progress_bar = tqdm(dataloader, desc="Train Epoch", leave=False)

    for inputs, labels, _ in progress_bar:
        # Move data to the specified device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Update progress bar description
        progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item() / labels.size(0):.2f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation data.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to evaluate on ('mps' or 'cpu').

    Returns:
        tuple: Average epoch loss (float), average epoch accuracy (float).
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Wrap dataloader with tqdm for a progress bar
    progress_bar = tqdm(dataloader, desc="Evaluate", leave=False)

    with torch.no_grad():  # Disable gradient calculations
        for inputs, labels, _ in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item() / labels.size(0):.2f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def train_and_evaluate(model, dataset, criterion, optimizer, epochs, device, config):
    """
    Trains a model for a specified number of epochs and returns the final
    training loss. Designed for quick particle evaluation in ComputeLoss.

    Args:
        model (nn.Module): The model to train.
        dataset: The DataLoader for the training data.
                 (Note: For ComputeLoss, this usually only contains training data).
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        epochs (int): Number of epochs to train.
        device (torch.device): Device ('mps' or 'cpu').
        config (dict): Configuration dictionary (currently unused here, but passed for flexibility).

    Returns:
        float: The average training loss from the *last* epoch.
               Returns float('inf') if training fails or results in invalid loss.
    """
    last_epoch_loss = float('inf')
    print(f"    Starting training for {epochs} epochs on device '{device}'...")
    train_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        # print(f"      Epoch {epoch + 1}/{epochs}") # Verbose epoch print

        # Perform one training epoch
        try:
            epoch_loss, epoch_acc = train_epoch(model, dataset, criterion, optimizer, device)
            last_epoch_loss = epoch_loss  # Store the loss from this epoch
            epoch_duration = time.time() - epoch_start_time
            print(
                f"      Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} (Duration: {epoch_duration:.2f}s)")

            # Handle potential issues during training
            if math.isnan(epoch_loss) or math.isinf(epoch_loss):
                print(
                    f"      Warning: Invalid loss ({epoch_loss}) encountered in epoch {epoch + 1}. Stopping training.")
                return float('inf')  # Return infinity if training produces invalid loss

        except Exception as e:
            import traceback
            print(f"      ERROR during training epoch {epoch + 1}: {e}")
            # traceback.print_exc() # Uncomment for detailed traceback
            return float('inf')  # Return infinity if any exception occurs during training

    train_duration = time.time() - train_start_time
    print(
        f"    Finished training {epochs} epochs (Total duration: {train_duration:.2f}s). Last epoch loss: {last_epoch_loss:.4f}")

    # Return the training loss from the last epoch as the fitness measure
    return last_epoch_loss


# --- Example Usage (Standalone Test) ---
if __name__ == '__main__':
    print("--- Testing utils/train.py ---")


    # Dummy Model
    class SimpleModel(nn.Module):
        def __init__(self, in_feat, out_feat):
            super().__init__()
            self.linear = nn.Linear(in_feat, out_feat)

        def forward(self, x):
            # Assume input is already flattened for simplicity
            return self.linear(x.view(x.size(0), -1))


    # Dummy Data
    # Create tensors directly on the target device
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    # Simulate batches of 10 samples, 3 channels, 8x8 images -> flattened 192 features
    dummy_features = 3 * 8 * 8
    dummy_num_classes = 5
    # Create dummy data and wrap in TensorDataset and DataLoader
    X_train = torch.randn(100, dummy_features, device=device)
    y_train = torch.randint(0, dummy_num_classes, (100,), device=device)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10)

    X_val = torch.randn(50, dummy_features, device=device)
    y_val = torch.randint(0, dummy_num_classes, (50,), device=device)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10)

    # Setup
    model = SimpleModel(dummy_features, dummy_num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    config = {}  # Empty config for this test

    # Test train_epoch
    print("\nTesting train_epoch...")
    loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"train_epoch Result - Loss: {loss:.4f}, Acc: {acc:.4f}")

    # Test evaluate_model
    print("\nTesting evaluate_model...")
    loss, acc = evaluate_model(model, val_loader, criterion, device)
    print(f"evaluate_model Result - Loss: {loss:.4f}, Acc: {acc:.4f}")

    # Test train_and_evaluate (used by ComputeLoss)
    print("\nTesting train_and_evaluate...")
    # Re-initialize model and optimizer for a clean test
    model = SimpleModel(dummy_features, dummy_num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    final_train_loss = train_and_evaluate(model, train_loader, criterion, optimizer, 3, device, config)
    print(f"train_and_evaluate Result (Final Train Loss): {final_train_loss:.4f}")

    print("\n--- Test Complete ---")
