# project_root/utils/helpers.py

import os
import json
import logging
import torch
import datetime


def setup_logging(log_file='experiment.log', level=logging.INFO):
    """
    Sets up basic logging to console and a file.

    Args:
        log_file (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # Append mode
            logging.StreamHandler()  # Log to console
        ]
    )
    logging.info("Logging setup complete.")
    print(f"Logging to console and file: {os.path.abspath(log_file)}")


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Saves model checkpoint including model state, optimizer state, epoch, and loss.

    Args:
        model (nn.Module): The PyTorch model.
        optimizer (optim.Optimizer): The optimizer.
        epoch (int): Current epoch number.
        loss (float): Current loss value (e.g., validation loss).
        filepath (str): Path where the checkpoint will be saved.
    """
    checkpoint_dir = os.path.dirname(filepath)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
            logging.info(f"Created checkpoint directory: {checkpoint_dir}")
        except OSError as e:
            logging.error(f"Error creating checkpoint directory {checkpoint_dir}: {e}")
            return  # Abort saving if directory creation fails

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    try:
        torch.save(state, filepath)
        logging.info(f"Checkpoint saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint to {filepath}: {e}")


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Loads model checkpoint. Optionally loads optimizer state.

    Args:
        filepath (str): Path to the checkpoint file.
        model (nn.Module): The PyTorch model instance (architecture must match).
        optimizer (optim.Optimizer, optional): The optimizer instance. Defaults to None.
        device (str or torch.device): Device to load the model onto. Defaults to 'cpu'.

    Returns:
        int: The epoch number saved in the checkpoint, or 0 if loading fails/no epoch saved.
        float: The loss value saved, or float('inf') if loading fails/no loss saved.
    """
    if not os.path.exists(filepath):
        logging.error(f"Checkpoint file not found: {filepath}")
        return 0, float('inf')

    try:
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)  # Ensure model is on the correct device after loading state

        start_epoch = checkpoint.get('epoch', 0)  # Default to 0 if epoch not saved
        saved_loss = checkpoint.get('loss', float('inf'))  # Default to Inf if loss not saved

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move optimizer state to the correct device (important if loading GPU checkpoint on CPU or vice-versa)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            logging.info(f"Loaded optimizer state from {filepath}")
        elif optimizer:
            logging.warning(
                f"Optimizer state not found in checkpoint or optimizer not provided. Optimizer not loaded: {filepath}")

        logging.info(
            f"Checkpoint loaded successfully from {filepath}. Resuming from epoch {start_epoch + 1}. Saved loss: {saved_loss:.4f}")
        return start_epoch, saved_loss

    except Exception as e:
        logging.error(f"Failed to load checkpoint from {filepath}: {e}")
        return 0, float('inf')


def save_results(results_dict, filepath):
    """
    Saves experiment results dictionary to a JSON file.

    Args:
        results_dict (dict): Dictionary containing experiment results.
        filepath (str): Path to the JSON file.
    """
    results_dir = os.path.dirname(filepath)
    if results_dir and not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
            logging.info(f"Created results directory: {results_dir}")
        except OSError as e:
            logging.error(f"Error creating results directory {results_dir}: {e}")
            return

    # Add timestamp to results
    results_dict['timestamp'] = datetime.datetime.now().isoformat()

    try:
        with open(filepath, 'w') as f:
            # Use indent for readability
            json.dump(results_dict, f, indent=4)
        logging.info(f"Results saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save results to {filepath}: {e}")


# --- Example Usage (Standalone Test) ---
if __name__ == '__main__':
    print("--- Testing utils/helpers.py ---")

    # Setup logging (creates a test log file)
    log_filename = "test_helpers.log"
    if os.path.exists(log_filename):
        os.remove(log_filename)  # Clean up previous test log
    setup_logging(log_filename, level=logging.DEBUG)
    logging.info("This is an info message.")
    logging.debug("This is a debug message.")

    # Test save/load checkpoint
    # Dummy model and optimizer
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())
    epoch = 5
    loss = 0.1234
    checkpoint_path = "test_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)  # Clean up previous checkpoint

    logging.info("\nTesting save_checkpoint...")
    save_checkpoint(model, optimizer, epoch, loss, checkpoint_path)

    logging.info("\nTesting load_checkpoint...")
    # Create new instances to load into
    new_model = torch.nn.Linear(10, 2)
    new_optimizer = torch.optim.Adam(new_model.parameters())
    loaded_epoch, loaded_loss = load_checkpoint(checkpoint_path, new_model, new_optimizer)
    print(f"Loaded epoch: {loaded_epoch}, Loaded loss: {loaded_loss}")
    # Simple check if loading worked (parameters should be the same)
    print(f"Model params match after load: {torch.equal(model.weight, new_model.weight)}")

    # Test save results
    logging.info("\nTesting save_results...")
    results = {
        'best_loss': 0.05,
        'best_accuracy': 0.98,
        'best_architecture': [
            {'type': 'conv', 'out_channels': 10, 'kernel_size': 3},
            {'type': 'fc', 'neurons': 2}
        ],
        'config': {'lr': 0.01, 'epochs': 10}
    }
    results_path = "test_results.json"
    if os.path.exists(results_path):
        os.remove(results_path)  # Clean up previous results

    save_results(results, results_path)
    # Verify file exists
    print(f"Results file exists: {os.path.exists(results_path)}")

    # Clean up test files
    # os.remove(log_filename)
    # os.remove(checkpoint_path)
    # os.remove(results_path)
    print("\n--- Test Complete (Check test_helpers.log, test_checkpoint.pth, test_results.json) ---")
