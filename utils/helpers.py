import argparse
import datetime
import json
import logging
import os
import sys
from datetime import datetime

import torch
from torch import nn, optim

# Global flag to track if logging has been configured
_logging_configured = False

def configure_logging(log_level='INFO'):
    """
    Configure logging in a way that works with multiprocessing.
    Uses a global flag to ensure logging is only configured once.

    Args:
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    global _logging_configured

    if _logging_configured:
        return

    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout  # Use stdout instead of stderr
    )

    # Set flag to prevent reconfiguration
    _logging_configured = True

    # Silence other loggers that might be too verbose
    logging.getLogger('PIL').setLevel(logging.WARNING)

    logging.debug("Logging configured successfully")


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
    results_dict['timestamp'] = datetime.now().isoformat()

    try:
        with open(filepath, 'w') as f:
            # Use indent for readability
            json.dump(results_dict, f, indent=4)
        logging.info(f"Results saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save results to {filepath}: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PSO-CNN Architecture Search for Action Recognition")

    # --- Paths ---
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Directory containing the action recognition dataset.")
    parser.add_argument('--output_dir', type=str, default="results",
                        help="Directory to save results, logs, and models.")

    # --- Experiment Setup ---
    parser.add_argument('--fusion_type', type=str, required=True, choices=['early', 'late'],
                        help="Fusion strategy to use.")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use ('mps', 'cuda', 'cpu'). Auto-detects if None.")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of dataloader workers.")
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help="Logging level.")

    # --- PSO Parameters ---
    parser.add_argument('--swarm_size', type=int, default=20, help="Number of particles in the swarm (N).")
    parser.add_argument('--max_iter', type=int, default=30, help="Maximum number of PSO iterations.")
    parser.add_argument('--max_layers', type=int, default=15, help="Maximum number of functional layers (l_max).")
    parser.add_argument('--max_fc_layers', type=int, default=5, help="Maximum number of fully connected layers.")
    parser.add_argument('--cg', type=float, default=0.7, help="PSO parameter Cg (gBest influence probability).")
    parser.add_argument('--k_max', type=int, default=7, help="Maximum Conv kernel size (odd number).")
    parser.add_argument('--maps_max', type=int, default=128, help="Maximum Conv feature maps.")
    parser.add_argument('--n_max', type=int, default=256, help="Maximum neurons in intermediate FC layers.")
    parser.add_argument('--n_out', type=int, default=20, help="Number of output classes (n_out).")

    # --- Training Parameters ---
    parser.add_argument('--e_train', type=int, default=50, help="Epochs for particle evaluation during PSO.")
    parser.add_argument('--e_test', type=int, default=10, help="Epochs for final training of the best model.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for Adam optimizer.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and evaluation.")

    # --- Optional Features ---
    parser.add_argument('--use_bn', action='store_true', help="Enable Batch Normalization in architectures.")
    parser.add_argument('--use_dropout', action='store_true', help="Enable Dropout in architectures.")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="Dropout probability if --use_dropout is set.")
    parser.add_argument('--log_dir', type=str, default="logs", help="Directory for experiment logs.")

    # --- PSO Options ---
    parser.add_argument('--use_multiprocessing', action='store_true',
                        help="Use multiprocessing PSO instead of standard PSO.")
    parser.add_argument('--num_processes', type=int, default=4,
                        help="Number of processes to use for multiprocessing PSO. Default: max(1, cpu_count() - 1)")

    return parser.parse_args()


def setup_experiment(args):
    """Set up the experiment configuration."""
    # Configure logging first
    configure_logging(args.log_level)

    # Create output directory if it doesn't exist
    logger = logging.getLogger(__name__)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"experiment_{args.fusion_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Determine device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Create experiment configuration
    config = {
        'data_dir': args.data_dir,
        'output_dir': output_dir,
        'fusion_type': args.fusion_type,
        'N': args.swarm_size,
        'iter_max': args.max_iter,
        'l_max': args.max_layers,
        'max_fc_layers': args.max_fc_layers,
        'Cg': args.cg,
        'k_max': args.k_max,
        'maps_max': args.maps_max,
        'n_max': args.n_max,
        'n_out': args.n_out,
        'e_train': args.e_train,
        'e_test': args.e_test,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'device': device,
        'use_bn': args.use_bn,
        'use_dropout': args.use_dropout,
        'dropout_rate': args.dropout_rate,
        'num_workers': args.num_workers,
        'use_multiprocessing': args.use_multiprocessing
    }

    # Add num_processes if specified
    if args.num_processes is not None:
        config['num_processes'] = args.num_processes

    # Save configuration to file
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w') as f:
        # Convert device to string for JSON serialization
        config_json = config.copy()
        config_json['device'] = str(config_json['device'])
        json.dump(config_json, f, indent=4)

    logger.info(f"Experiment configuration saved to {config_file}")

    return config
