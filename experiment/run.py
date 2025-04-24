import argparse
import json
import logging
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_preprocessing.dataset import ActionDataset
from models.fusion_models import build_early_fusion_model, build_late_fusion_model
from pso.pso import psoCNN
from pso.multiprocessing_pso import multiprocessing_psoCNN
from utils.helpers import save_results
from utils.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_metrics_history,
    MetricsLogger
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    parser.add_argument('--device', type=str, default=None, help="Device to use ('cuda', 'cpu'). Auto-detects if None.")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help="Logging level.")

    # --- PSO Parameters ---
    parser.add_argument('--swarm_size', type=int, default=20, help="Number of particles in the swarm (N).")
    parser.add_argument('--max_iter', type=int, default=30, help="Maximum number of PSO iterations.")
    parser.add_argument('--max_layers', type=int, default=15, help="Maximum number of functional layers (l_max).")
    parser.add_argument('--cg', type=float, default=0.7, help="PSO parameter Cg (gBest influence probability).")
    parser.add_argument('--k_max', type=int, default=7, help="Maximum Conv kernel size (odd number).")
    parser.add_argument('--maps_max', type=int, default=128, help="Maximum Conv feature maps.")
    parser.add_argument('--n_max', type=int, default=256, help="Maximum neurons in intermediate FC layers.")
    parser.add_argument('--n_out', type=int, default=20, help="Number of output classes (n_out).")

    # --- Training Parameters ---
    parser.add_argument('--e_train', type=int, default=5, help="Epochs for particle evaluation during PSO.")
    parser.add_argument('--e_test', type=int, default=50, help="Epochs for final training of the best model.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for Adam optimizer.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and evaluation.")

    # --- Optional Features ---
    parser.add_argument('--use_bn', action='store_true', help="Enable Batch Normalization in architectures.")
    parser.add_argument('--use_dropout', action='store_true', help="Enable Dropout in architectures.")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="Dropout probability if --use_dropout is set.")
    parser.add_argument('--log_dir', type=str, default="logs", help="Directory for experiment logs.")

    # --- PSO Options ---
    parser.add_argument('--use_multiprocessing', action='store_true', help="Use multiprocessing PSO instead of standard PSO.")
    parser.add_argument('--num_processes', type=int, default=None, 
                        help="Number of processes to use for multiprocessing PSO. Default: max(1, cpu_count() - 1)")

    return parser.parse_args()


def setup_experiment(args):
    """Set up the experiment configuration."""
    # Create output directory if it doesn't exist
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
        'Cg': args.cg,
        'k_max': args.k_max,
        'maps_max': args.maps_max,
        'n_max': args.n_max,
        'n_out': args.n_out,  # To be set based on dataset
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


# --- Main Experiment Function ---

def run_pso_experiment(config):
    """
    Runs a full PSO-CNN experiment based on the provided configuration.

    Args:
        config (dict): Dictionary containing all experiment parameters.
    """
    experiment_start_time = time.time()
    logging.info("=" * 30 + " Starting Experiment " + "=" * 30)
    logging.info(f"Configuration:\n{config}")

    # --- Setup Device ---
    device = config.get('device', torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    logging.info(f"Using device: {device}")

    # --- Load Data ---
    logging.info("Loading dataset...")
    try:
        # Create datasets for train, validation (optional), and test
        # The dataset needs to handle returning data formatted for the specific fusion_type
        train_dataset = ActionDataset(config['data_dir'], split='train', fusion_type=config['fusion_type'])
        val_dataset = ActionDataset(config['data_dir'], split='val', fusion_type=config['fusion_type'])  # Optional
        test_dataset = ActionDataset(config['data_dir'], split='test', fusion_type=config['fusion_type'])

        # Get data details (channels, dimensions, classes)
        # The dataset class should provide a method for this
        input_channels, input_height, input_width, num_classes = train_dataset.get_details()
        logging.info(
            f"Dataset details: Input Channels={input_channels}, H={input_height}, W={input_width}, Num Classes={num_classes}")

        # Update config with dataset details if needed by PSO/model builders
        config['input_channels'] = input_channels  # Might be tuple for late fusion
        config['input_height'] = input_height
        config['input_width'] = input_width
        config['num_classes'] = num_classes
        config['n_out'] = num_classes  # Ensure n_out matches dataset

        # Create DataLoaders
        batch_size = config.get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=config.get('num_workers', 2))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=config.get('num_workers', 2))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=config.get('num_workers', 2))

    except FileNotFoundError:
        logging.error(f"Data directory not found: {config['data_dir']}")
        return
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Run PSO Architecture Search ---
    logging.info("Starting PSO architecture search...")
    # Prepare config for PSO (pass train_loader as the 'dataset' for evaluation)
    pso_config = config.copy()
    pso_config['dataset'] = train_loader  # Use train loader for particle fitness eval
    # pso_config['dataset'] = (train_loader, val_loader) # Or pass both if ComputeLoss uses validation

    # Choose between standard PSO and multiprocessing PSO
    if config.get('use_multiprocessing', False):
        logging.info("Using multiprocessing PSO for architecture search")
        if 'num_processes' in config:
            logging.info(f"Using {config['num_processes']} processes")
        best_particle, pso_best_loss = multiprocessing_psoCNN(pso_config)
    else:
        logging.info("Using standard PSO for architecture search")
        best_particle, pso_best_loss = psoCNN(pso_config)

    if best_particle is None:
        logging.error("PSO search failed to find a valid architecture.")
        return

    logging.info(f"PSO search finished. Best particle loss during search: {pso_best_loss:.4f}")
    logging.info(f"Best architecture found (length {len(best_particle.architecture)}):")
    # Log the best architecture found
    for l_idx, layer in enumerate(best_particle.architecture): logging.info(f"  L{l_idx + 1}: {layer}")

    # --- Final Model Training & Evaluation ---
    logging.info("Starting final training of the best architecture...")
    final_model = None
    try:
        # Build the final model based on the fusion type and best architecture
        fusion_type = config['fusion_type']
        if fusion_type == 'early':
            # Early fusion uses a single architecture
            final_model = build_early_fusion_model(
                best_particle.architecture,
                input_channels,  # Should be combined channels for early fusion
                num_classes,
                config
            )
        elif fusion_type == 'late':
            # Late fusion might have one or two architectures in the particle
            # Assuming particle.architecture holds BOTH if optimized jointly,
            # or just one if optimized separately/identically. Adapt as needed.
            # This example assumes one architecture applied to both branches.
            # Modify if your Particle encodes two architectures for late fusion.
            if not isinstance(input_channels, tuple) or len(input_channels) != 2:
                raise ValueError("Late fusion requires input_channels to be a tuple (color_channels, depth_channels)")
            final_model = build_late_fusion_model(
                best_particle.architecture,  # Arch for color branch
                best_particle.architecture,  # Arch for depth branch (or use a second arch if available)
                input_channels[0],  # Color channels
                input_channels[1],  # Depth channels
                num_classes,
                config
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        final_model.to(device)

        # Set up final training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(final_model.parameters(), lr=config.get('learning_rate', 0.001))
        epochs = config['e_test']

        # Create TensorBoard log directory for training
        train_log_dir = os.path.join(config['output_dir'],
                                     f"tensorboard_training_{fusion_type}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(train_log_dir, exist_ok=True)

        # Initialize metrics logger for training
        train_metrics_logger = MetricsLogger(train_log_dir)

        # Train the model with more comprehensive metrics tracking
        logging.info(f"Training final model for {epochs} epochs...")

        # Initialize metrics history for plotting
        metrics_history = {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }

        # Training loop with metrics tracking
        for epoch in range(epochs):
            epoch_start_time = time.time()

            # Train for one epoch
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, data in enumerate(train_loader):
                if len(data) == 2:  # Early fusion
                    inputs, targets = data
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                else:  # Late fusion
                    color_inputs, depth_inputs, targets = data
                    color_inputs, depth_inputs, targets = color_inputs.to(device), depth_inputs.to(device), targets.to(
                        device)

                    optimizer.zero_grad()
                    outputs = model(color_inputs, depth_inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            # Calculate training metrics for this epoch
            train_loss = train_loss / len(train_loader)
            train_acc = correct / total

            # Evaluate on validation set (using test set as validation for simplicity)
            val_metrics = calculate_metrics(model, test_loader, criterion, device)

            # Log metrics
            epoch_metrics = {
                'loss': train_loss,
                'accuracy': train_acc,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1_score': val_metrics['f1_score']
            }

            # Log to TensorBoard
            train_metrics_logger.log_metrics(epoch_metrics, epoch)

            # Store metrics for history plotting
            metrics_history['loss'].append(train_loss)
            metrics_history['accuracy'].append(train_acc)
            metrics_history['precision'].append(val_metrics['precision'])
            metrics_history['recall'].append(val_metrics['recall'])
            metrics_history['f1_score'].append(val_metrics['f1_score'])

            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            logging.info(f"Epoch {epoch + 1}/{epochs} - "
                         f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                         f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                         f"Val F1: {val_metrics['f1_score']:.4f}, "
                         f"Time: {epoch_time:.2f}s")

        # Close the training metrics logger
        train_metrics_logger.close()

        # Create visualization directory for training history
        train_vis_dir = os.path.join(config['output_dir'],
                                     f"training_history_{fusion_type}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(train_vis_dir, exist_ok=True)

        # Plot training history
        plot_metrics_history(metrics_history, train_vis_dir)

        # Final training loss for compatibility with existing code
        final_train_loss = train_loss
        logging.info(f"Final model training completed. Last epoch training loss: {final_train_loss:.4f}")

        # Evaluate on the TEST set
        logging.info("Evaluating final model on test set...")
        final_metrics = calculate_metrics(model, test_loader, criterion, device)
        logging.info(
            f"Final Test Set Performance - Loss: {final_metrics['loss']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")

        # --- Calculate Comprehensive Metrics ---
        logging.info("Calculating comprehensive evaluation metrics...")

        # Create output directories for visualizations
        vis_dir = os.path.join(config['output_dir'], f"visualizations_{fusion_type}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(vis_dir, exist_ok=True)

        # Create TensorBoard log directory
        log_dir = os.path.join(config['output_dir'], f"tensorboard_{fusion_type}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)

        # Initialize metrics logger
        metrics_logger = MetricsLogger(log_dir)

        # Calculate comprehensive metrics
        metrics = calculate_metrics(final_model, test_loader, criterion, device)

        # Log metrics to console
        logging.info("Comprehensive Evaluation Metrics:")
        logging.info(f"  Loss: {metrics['loss']:.4f}")
        logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"  Precision: {metrics['precision']:.4f}")
        logging.info(f"  Recall: {metrics['recall']:.4f}")
        logging.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logging.info(f"  Model Parameters: {metrics['parameters']:,}")
        logging.info(f"  Model Size: {metrics['model_size_mb']:.2f} MB")
        logging.info(f"  Inference Time: {metrics['inference_time_ms']:.2f} ms")
        logging.info(f"  FLOPs: {metrics['flops']:,}")

        # Log metrics to TensorBoard
        metrics_logger.log_metrics(metrics, 0)  # 0 for final evaluation
        metrics_logger.close()

        # Plot confusion matrix
        class_names = [str(i) for i in range(num_classes)]  # Generate class names
        plot_confusion_matrix(metrics['confusion_matrix'], class_names, vis_dir)

        # --- Save Results ---
        results = {
            'config': config,
            'best_particle_loss_pso': pso_best_loss,
            'best_architecture': best_particle.architecture,
            'final_model_train_loss': final_train_loss,
            'final_model_test_loss': metrics['loss'],
            'final_model_test_accuracy': metrics['accuracy'],
            'final_model_test_precision': metrics['precision'],
            'final_model_test_recall': metrics['recall'],
            'final_model_test_f1_score': metrics['f1_score'],
            'final_model_parameters': metrics['parameters'],
            'final_model_size_mb': metrics['model_size_mb'],
            'final_model_inference_time_ms': metrics['inference_time_ms'],
            'final_model_flops': metrics['flops'],
            'experiment_duration_seconds': time.time() - experiment_start_time,
        }
        results_filename = f"results_{fusion_type}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        results_filepath = os.path.join(config['output_dir'], results_filename)
        save_results(results, results_filepath)

        # --- Save Final Model (Optional) ---
        model_filename = f"final_model_{fusion_type}_{time.strftime('%Y%m%d_%H%M%S')}.pth"
        model_filepath = os.path.join(config['output_dir'], model_filename)
        # Save model state only for inference, or full checkpoint if needed
        torch.save(final_model.state_dict(), model_filepath)
        logging.info(f"Final model state saved to {model_filepath}")
        # save_checkpoint(final_model, optimizer, epochs, test_loss, model_filepath.replace('.pth', '_ckpt.pth')) # Save full checkpoint


    except Exception as e:
        logging.error(f"Error during final training or evaluation: {e}", exc_info=True)

    logging.info("=" * 30 + " Experiment Finished " + "=" * 30)
    experiment_duration = time.time() - experiment_start_time
    logging.info(f"Total experiment duration: {experiment_duration:.2f} seconds")


# --- Example Usage (if run directly, though usually called from main.py) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    print("--- Running experiments/run_experiment.py Standalone Test ---")

    # Create dummy directories and config for testing
    if not os.path.exists("test_output"): os.makedirs("test_output")
    if not os.path.exists("test_data"): os.makedirs("test_data")  # Dummy data dir

    test_config = {
        'data_dir': 'test_data',
        'output_dir': 'test_output',
        'fusion_type': 'early',  # 'early' or 'late'
        'N': 4,  # Swarm size
        'iter_max': 2,  # Max iterations
        'l_max': 5,  # Max functional layers
        'Cg': 0.7,  # gBest probability factor
        'k_max': 3,  # Max kernel size (e.g., 3x3)
        'maps_max': 8,  # Max feature maps per conv layer
        'n_max': 16,  # Max neurons per intermediate FC layer
        # n_out, input_channels, num_classes determined from dummy dataset
        'e_train': 1,  # Epochs for particle evaluation (quick)
        'e_test': 2,  # Epochs for final training
        'learning_rate': 0.005,
        'batch_size': 8,
        'device': torch.device("mps" if torch.mps.is_available() else "cpu"),
        'use_bn': True,
        'use_dropout': False,
        'dropout_rate': 0.5,
        'num_workers': 0  # Avoid multiprocessing issues in simple tests
    }

    run_pso_experiment(test_config)

    print("\n--- Standalone Test Complete ---")
