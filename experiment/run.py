import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from models.fusion_models import build_early_fusion_model
from pso.multiprocessing_pso import multiprocessing_psoCNN
from pso.pso import psoCNN
from utils.helpers import save_results, parse_args, setup_experiment
from utils.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_metrics_history,
    MetricsLogger
)

# Get logger for this module
logger = logging.getLogger(__name__)


# --- Training Function for Late Fusion Models ---

def train_model(model_type, model, train_loader, val_loader, device, epochs, learning_rate, config):
    """
    Trains a model for late fusion.

    Args:
        model_type (str): Type of model ('color' or 'depth')
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to train on
        epochs (int): Number of epochs to train for
        learning_rate (float): Learning rate for optimizer
        config (dict): Configuration dictionary

    Returns:
        dict: Metrics history
    """
    logging.info(f"Training {model_type} stream model for {epochs} epochs...")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Create TensorBoard log directory
    log_dir = os.path.join(config['output_dir'],
                          f"tensorboard_training_{model_type}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    metrics_logger = MetricsLogger(log_dir)

    # Initialize metrics history
    metrics_history = {
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }

    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Train for one epoch
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, action_labels, _) in enumerate(train_loader):
            inputs, action_labels = inputs.to(device), action_labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, action_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += action_labels.size(0)
            correct += predicted.eq(action_labels).sum().item()

        # Calculate training metrics for this epoch
        train_loss = train_loss / len(train_loader)
        train_acc = correct / total

        # Evaluate on validation set
        val_metrics = calculate_metrics(model, val_loader, criterion, device)

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
        metrics_logger.log_metrics(epoch_metrics, epoch)

        # Store metrics for history plotting
        metrics_history['loss'].append(train_loss)
        metrics_history['accuracy'].append(train_acc)
        metrics_history['precision'].append(val_metrics['precision'])
        metrics_history['recall'].append(val_metrics['recall'])
        metrics_history['f1_score'].append(val_metrics['f1_score'])

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        logging.info(f"{model_type.capitalize()} Model - Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Val F1: {val_metrics['f1_score']:.4f}, "
                    f"Time: {epoch_time:.2f}s")

    # Close the metrics logger
    metrics_logger.close()

    # Create visualization directory for training history
    vis_dir = os.path.join(config['output_dir'],
                          f"training_history_{model_type}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(vis_dir, exist_ok=True)

    # Plot training history
    plot_metrics_history(metrics_history, vis_dir)

    return metrics_history


# --- Main Experiment Function ---

def setup_dataloaders(config):
    """
    Sets up the dataloaders for the experiment based on the fusion type.

    Args:
        config (dict): Dictionary containing experiment parameters.

    Returns:
        tuple: Dataloaders and dataset info, or None if there's an error.
    """
    logging.info("Loading dataset...")
    try:
        # Use the get_dataloaders function to create DataLoaders for train, validation, and test
        from data_preprocessing.dataset import get_dataloaders

        # Handle different return formats based on fusion type
        fusion_type = config.get('fusion_type')
        if fusion_type == 'early':
            train_loader, val_loader, test_loader, dataset_info = get_dataloaders(config)
            # For PSO, we only need the train loader
            pso_dataset = train_loader

            # Create a dictionary to store all loaders
            loaders = {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'pso_dataset': pso_dataset
            }
        else:  # late fusion
            # For late fusion, we get separate color and depth loaders
            (color_train_loader, color_val_loader, color_test_loader,
             depth_train_loader, depth_val_loader, depth_test_loader,
             dataset_info) = get_dataloaders(config)

            # For PSO, we'll use the color loader (arbitrary choice, could use either)
            pso_dataset = color_train_loader

            # Create a dictionary to store all loaders
            loaders = {
                'color_train_loader': color_train_loader,
                'color_val_loader': color_val_loader,
                'color_test_loader': color_test_loader,
                'depth_train_loader': depth_train_loader,
                'depth_val_loader': depth_val_loader,
                'depth_test_loader': depth_test_loader,
                'pso_dataset': pso_dataset
            }

        # Extract dataset details
        input_channels = dataset_info['input_channels']
        input_height = dataset_info['img_height']
        input_width = dataset_info['img_width']
        num_action_classes = dataset_info['num_action_classes']
        num_tool_classes = dataset_info['num_tool_classes']

        logging.info(
            f"Dataset details: Input Channels={input_channels}, H={input_height}, W={input_width}, "
            f"Action Classes={num_action_classes}, Tool Classes={num_tool_classes}")

        # Create a dictionary to store dataset info
        dataset_details = {
            'input_channels': input_channels,
            'input_height': input_height,
            'input_width': input_width,
            'num_action_classes': num_action_classes,
            'num_tool_classes': num_tool_classes
        }

        return loaders, dataset_details

    except FileNotFoundError:
        logging.error(f"Data directory not found: {config['data_dir']}")
        return None, None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def train_early_fusion_model(best_particle, config, loaders, device):
    """
    Builds and trains the early fusion model using the best architecture found by PSO.

    Args:
        best_particle: The particle with the best architecture found by PSO.
        config (dict): Dictionary containing experiment parameters.
        loaders (dict): Dictionary containing the dataloaders.
        device: Device to train on.

    Returns:
        tuple: Trained model and metrics history.
    """
    logging.info("Building early fusion model...")
    input_channels = config['input_channels']
    num_action_classes = config['num_action_classes']
    epochs = config['e_test']

    final_model = build_early_fusion_model(
        best_particle.architecture,
        input_channels,  # Combined channels for early fusion
        num_action_classes,  # Using action classes for classification
        config
    )
    final_model.to(device)

    # Set up optimizer and criterion
    optimizer = optim.Adam(final_model.parameters(), lr=config.get('learning_rate', 0.0001))
    criterion = nn.CrossEntropyLoss()

    # Create TensorBoard log directory for training
    train_log_dir = os.path.join(config['output_dir'],
                                 f"tensorboard_training_early_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(train_log_dir, exist_ok=True)

    # Initialize metrics logger for training
    train_metrics_logger = MetricsLogger(train_log_dir)

    # Train the model with comprehensive metrics tracking
    logging.info(f"Training early fusion model for {epochs} epochs...")

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
        final_model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, action_labels, _) in enumerate(loaders['train_loader']):
            inputs, action_labels = inputs.to(device), action_labels.to(device)

            optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, action_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += action_labels.size(0)
            correct += predicted.eq(action_labels).sum().item()

        # Calculate training metrics for this epoch
        train_loss = train_loss / len(loaders['train_loader'])
        train_acc = correct / total

        # Evaluate on validation set
        val_metrics = calculate_metrics(final_model, loaders['val_loader'], criterion, device)

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
                                 f"training_history_early_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(train_vis_dir, exist_ok=True)

    # Plot training history
    plot_metrics_history(metrics_history, train_vis_dir)

    logging.info(f"Early fusion model training completed. Last epoch training loss: {train_loss:.4f}")

    return final_model, metrics_history, train_loss


def train_late_fusion_models(config, device):
    """
    Runs PSO search for color and depth models, then builds and trains them.

    Args:
        config (dict): Dictionary containing experiment parameters.
        device: Device to train on.

    Returns:
        tuple: Trained color and depth models, and metrics histories.
    """
    input_channels = config['input_channels']
    num_action_classes = config['num_action_classes']
    epochs = config['e_test']

    if not isinstance(input_channels, tuple) or len(input_channels) != 2:
        raise ValueError("Late fusion requires input_channels to be a tuple (color_channels, depth_channels)")

    # Run PSO for color model first
    logging.info("Starting PSO architecture search for COLOR model...")
    # Prepare config for color PSO
    color_pso_config = config.copy()
    color_pso_config['dataset'] = config['color_train_loader']  # Use color dataset
    color_pso_config['input_channels'] = input_channels[0]  # Use color channels only

    # Run PSO for color model
    color_best_particle, color_pso_best_loss = run_pso_search(color_pso_config, config['color_train_loader'])

    if color_best_particle is None:
        logging.error("PSO search failed to find a valid architecture for color model.")
        return None, None, None, None, float('inf')

    # Build color model with best architecture
    logging.info("Building color stream model...")
    model_color = build_early_fusion_model(
        color_best_particle.architecture,  # Use color-specific architecture
        input_channels[0],  # Color channels
        num_action_classes,
        config
    )
    model_color.to(device)

    # Run PSO for depth model
    logging.info("Starting PSO architecture search for DEPTH model...")
    # Prepare config for depth PSO
    depth_config = config.copy()
    depth_config['dataset'] = config['depth_train_loader']  # Use depth dataset
    depth_config['input_channels'] = input_channels[1]  # Use depth channels only
    depth_config['activation_type'] = 'leaky_relu'
    depth_config['leaky_relu_slope'] = 0.3  # Set slope parameter for leaky ReLU

    # Run PSO for depth model
    depth_best_particle, depth_pso_best_loss = run_pso_search(depth_config, config['depth_train_loader'])

    if depth_best_particle is None:
        logging.error("PSO search failed to find a valid architecture for depth model.")
        return None, None, None, None, float('inf')

    # Build depth model with best architecture
    logging.info("Building depth stream model...")
    model_depth = build_early_fusion_model(
        depth_best_particle.architecture,  # Use depth-specific architecture
        input_channels[1],  # Depth channels
        num_action_classes,
        depth_config
    )
    model_depth.to(device)

    # Train both models sequentially instead of in parallel to avoid multiprocessing issues
    logging.info("Starting sequential training of color and depth models...")

    # Train color model first
    logging.info("Training color model...")
    color_metrics_history = train_model('color', model_color, config['color_train_loader'], 
                                       config['color_val_loader'], device, epochs, 
                                       config.get('learning_rate', 0.0001), config)

    # Train depth model next
    logging.info("Training depth model...")
    depth_metrics_history = train_model('depth', model_depth, config['depth_train_loader'], 
                                      config['depth_val_loader'], device, epochs, 
                                      config.get('learning_rate', 0.0001), config)

    logging.info("Sequential training of color and depth models completed.")

    # Final training loss for compatibility with existing code (average of both models)
    final_train_loss = (depth_metrics_history['loss'][-1] + color_metrics_history['loss'][-1]) / 2
    logging.info(
        f"Late fusion models training completed. Last epoch average training loss: {final_train_loss:.4f}")

    return model_color, model_depth, color_metrics_history, depth_metrics_history, final_train_loss


def evaluate_early_fusion_model(model, loaders, config, device):
    """
    Evaluates the early fusion model on the test set.

    Args:
        model: The trained early fusion model.
        loaders (dict): Dictionary containing the dataloaders.
        config (dict): Dictionary containing experiment parameters.
        device: Device to evaluate on.

    Returns:
        dict: Evaluation metrics.
    """
    criterion = nn.CrossEntropyLoss()

    logging.info("Evaluating early fusion model on test set...")
    metrics = calculate_metrics(model, loaders['test_loader'], criterion, device)
    logging.info(
        f"Final Test Set Performance - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    # Create output directories for visualizations
    vis_dir = os.path.join(config['output_dir'],
                           f"visualizations_early_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(vis_dir, exist_ok=True)

    # Create TensorBoard log directory
    log_dir = os.path.join(config['output_dir'], f"tensorboard_early_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    # Initialize metrics logger
    metrics_logger = MetricsLogger(log_dir)

    # Log metrics to TensorBoard
    metrics_logger.log_metrics(metrics, 0)  # 0 for final evaluation
    metrics_logger.close()

    # Plot confusion matrix
    # Use action class names (0, 1, 2, 3) corresponding to the actions
    action_names = ["push", "pull", "left_to_right", "right_to_left"]
    class_names = [f"{i}: {action}" for i, action in enumerate(action_names)]
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, vis_dir)

    return metrics


def evaluate_late_fusion_models(model_color, model_depth, config, device):
    """
    Evaluates the late fusion models (color and depth) on the test set.

    Args:
        model_color: The trained color model.
        model_depth: The trained depth model.
        config (dict): Dictionary containing experiment parameters.
        device: Device to evaluate on.

    Returns:
        dict: Evaluation metrics.
    """
    criterion = nn.CrossEntropyLoss()

    logging.info("Evaluating decision fusion on test set...")
    from utils.metrics import evaluate_decision_fusion
    metrics = evaluate_decision_fusion(model_color, model_depth, config['color_test_loader'], 
                                      config['depth_test_loader'], criterion, device)
    logging.info(
        f"Final Decision Fusion Test Performance - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    # Create output directories for visualizations
    vis_dir = os.path.join(config['output_dir'],
                           f"visualizations_late_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(vis_dir, exist_ok=True)

    # Create TensorBoard log directory
    log_dir = os.path.join(config['output_dir'], f"tensorboard_late_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    # Initialize metrics logger
    metrics_logger = MetricsLogger(log_dir)

    # Log metrics to TensorBoard
    metrics_logger.log_metrics(metrics, 0)  # 0 for final evaluation
    metrics_logger.close()

    # Plot confusion matrix
    # Use action class names (0, 1, 2, 3) corresponding to the actions
    action_names = ["push", "pull", "left_to_right", "right_to_left"]
    class_names = [f"{i}: {action}" for i, action in enumerate(action_names)]
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, vis_dir)

    return metrics


def save_experiment_results(config, best_particle, pso_best_loss, final_train_loss, metrics, 
                           model=None, model_color=None, model_depth=None, experiment_start_time=None):
    """
    Saves the experiment results, including metrics and models.

    Args:
        config (dict): Dictionary containing experiment parameters.
        best_particle: The particle with the best architecture found by PSO.
        pso_best_loss (float): The loss of the best architecture found by PSO.
        final_train_loss (float): The final training loss.
        metrics (dict): Evaluation metrics.
        model: The trained early fusion model (if applicable).
        model_color: The trained color model (if applicable).
        model_depth: The trained depth model (if applicable).
        experiment_start_time (float): The start time of the experiment.

    Returns:
        None
    """
    fusion_type = config['fusion_type']

    # Prepare results dictionary
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
    }

    # Add experiment duration if start time is provided
    if experiment_start_time is not None:
        results['experiment_duration_seconds'] = time.time() - experiment_start_time

    # Save results to file
    results_filename = f"results_{fusion_type}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    results_filepath = os.path.join(config['output_dir'], results_filename)
    save_results(results, results_filepath)

    # Save models
    if fusion_type == 'early' and model is not None:
        model_filename = f"final_model_{fusion_type}_{time.strftime('%Y%m%d_%H%M%S')}.pth"
        model_filepath = os.path.join(config['output_dir'], model_filename)
        torch.save(model.state_dict(), model_filepath)
        logging.info(f"Final model state saved to {model_filepath}")
    elif fusion_type == 'late':
        if model_color is not None:
            color_model_filename = f"final_model_color_{time.strftime('%Y%m%d_%H%M%S')}.pth"
            color_model_filepath = os.path.join(config['output_dir'], color_model_filename)
            torch.save(model_color.state_dict(), color_model_filepath)
            logging.info(f"Color model state saved to {color_model_filepath}")

        if model_depth is not None:
            depth_model_filename = f"final_model_depth_{time.strftime('%Y%m%d_%H%M%S')}.pth"
            depth_model_filepath = os.path.join(config['output_dir'], depth_model_filename)
            torch.save(model_depth.state_dict(), depth_model_filepath)
            logging.info(f"Depth model state saved to {depth_model_filepath}")


def run_pso_search(config, pso_dataset):
    """
    Runs the PSO architecture search to find the best architecture.

    Args:
        config (dict): Dictionary containing experiment parameters.
        pso_dataset: Dataset to use for PSO evaluation.

    Returns:
        tuple: Best particle and its loss, or (None, float('inf')) if there's an error.
    """
    logging.info("Starting PSO architecture search...")
    # Prepare config for PSO (pass appropriate dataset for evaluation)
    pso_config = config.copy()
    pso_config['dataset'] = pso_dataset  # Use the dataset selected based on fusion type
    # pso_config['dataset'] = (pso_dataset, val_dataset) # Or pass both if ComputeLoss uses validation

    # Choose between standard PSO and multiprocessing PSO
    if config.get('use_multiprocessing', True):
        logging.info("Using multiprocessing PSO for architecture search")
        if 'num_processes' in config:
            logging.info(f"Using {config['num_processes']} processes")
        best_particle, pso_best_loss = multiprocessing_psoCNN(pso_config)
    else:
        logging.info("Using standard PSO for architecture search")
        best_particle, pso_best_loss = psoCNN(pso_config)

    if best_particle is None:
        logging.error("PSO search failed to find a valid architecture.")
        return None, float('inf')

    logging.info(f"PSO search finished. Best particle loss during search: {pso_best_loss:.4f}")
    logging.info(f"Best architecture found (length {len(best_particle.architecture)}):")
    # Log the best architecture found
    for l_idx, layer in enumerate(best_particle.architecture): 
        logging.info(f"  L{l_idx + 1}: {layer}")

    return best_particle, pso_best_loss


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
    loaders, dataset_details = setup_dataloaders(config)
    if loaders is None or dataset_details is None:
        return

    # Update config with dataset details if needed by PSO/model builders
    config['input_channels'] = dataset_details['input_channels']  # Might be tuple for late fusion
    config['input_height'] = dataset_details['input_height']
    config['input_width'] = dataset_details['input_width']
    config['num_action_classes'] = dataset_details['num_action_classes']
    config['num_tool_classes'] = dataset_details['num_tool_classes']
    config['num_classes'] = dataset_details['num_action_classes']  # Using action classes for classification
    config['n_out'] = dataset_details['num_action_classes']  # Ensure n_out matches dataset

    # Store loaders in config for late fusion
    if config.get('fusion_type') == 'late':
        config['color_train_loader'] = loaders['color_train_loader']
        config['color_val_loader'] = loaders['color_val_loader']
        config['color_test_loader'] = loaders['color_test_loader']
        config['depth_train_loader'] = loaders['depth_train_loader']
        config['depth_val_loader'] = loaders['depth_val_loader']
        config['depth_test_loader'] = loaders['depth_test_loader']

    try:
        # --- Run PSO Architecture Search ---
        fusion_type = config['fusion_type']

        if fusion_type == 'early':
            # --- Early Fusion: Single Model ---
            # Run PSO to find the best architecture
            best_particle, pso_best_loss = run_pso_search(config, loaders['pso_dataset'])
            if best_particle is None:
                return

            # Train the early fusion model
            final_model, metrics_history, final_train_loss = train_early_fusion_model(
                best_particle, config, loaders, device)

            # Evaluate the early fusion model
            metrics = evaluate_early_fusion_model(final_model, loaders, config, device)

            # Save results and model
            save_experiment_results(
                config, best_particle, pso_best_loss, final_train_loss, metrics,
                model=final_model, experiment_start_time=experiment_start_time)

        elif fusion_type == 'late':
            # --- Late Fusion: Two Separate Models ---
            # Train the late fusion models (includes PSO search for each model)
            model_color, model_depth, color_metrics_history, depth_metrics_history, final_train_loss = train_late_fusion_models(
                config, device)

            if model_color is None or model_depth is None:
                return

            # Evaluate the late fusion models
            metrics = evaluate_late_fusion_models(model_color, model_depth, config, device)

            # For late fusion, we use the color model's best particle for results
            # (This is a simplification; in reality, we have two best particles)
            color_pso_config = config.copy()
            color_pso_config['dataset'] = config['color_train_loader']
            color_pso_config['input_channels'] = config['input_channels'][0]
            best_particle, _ = run_pso_search(color_pso_config, config['color_train_loader'])

            if best_particle is None:
                return

            # Save results and models
            save_experiment_results(
                config, best_particle, 0.0, final_train_loss, metrics,
                model_color=model_color, model_depth=model_depth, 
                experiment_start_time=experiment_start_time)

        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    except Exception as e:
        logging.error(f"Error during experiment: {e}", exc_info=True)

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
