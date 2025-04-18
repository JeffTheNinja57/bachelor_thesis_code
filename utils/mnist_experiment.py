# project_root/experiments/mnist_experiment.py

import os
import logging
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse  # Use argparse directly here for simplicity

# --- Project Imports ---
# Assume these modules exist and are importable relative to project_root
# Adjust paths if your execution context is different
try:
    from data_preprocessing.dataset import ActionDataset  # Using the MNIST simulating version
    from models.cnn_architecture import build_cnn  # Direct CNN builder
    from pso.pso import psoCNN
    from utils.helpers import save_results, setup_logging  # Assumes helpers.py exists
    from utils.train import train_and_evaluate, evaluate_model  # Assumes train.py exists
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}. Ensure all required files exist and PYTHONPATH is set.")


    # Define dummy placeholders if imports fail, to allow script structure check
    class ActionDataset:
        def __init__(self, data_dir, split='train',
                     fusion_type='early'): self.len = 100; self.fusion_type = fusion_type; self.channels = 4; self.h = 28; self.w = 28; self.classes = 10  # Simulate early fusion output

        def __len__(self): return self.len

        def __getitem__(self, idx): return torch.randn(self.channels, self.h, self.w), idx % self.classes

        def get_details(self): return self.channels, self.h, self.w, self.classes


    def build_cnn(arch, in_c, n_cls, cfg):
        print("DUMMY: Build CNN"); return nn.Linear(10, n_cls)  # Minimal dummy


    def psoCNN(cfg):
        print("DUMMY: psoCNN called"); return (
        type('obj', (object,), {'architecture': [{'type': 'fc', 'neurons': cfg['n_out']}]})(),
        0.5)  # Dummy best particle + loss


    def save_results(res, fp):
        print(f"DUMMY: Save results to {fp}")


    def setup_logging(log_file, level):
        print(f"DUMMY: Setup logging to {log_file}")


    def train_and_evaluate(model, dataset, criterion, optimizer, epochs, device, config):
        print("DUMMY: train_and_evaluate called"); return 0.5  # Dummy loss


    def evaluate_model(model, dataset, criterion, device):
        print("DUMMY: evaluate_model called"); return 0.4, 0.9  # Dummy loss, acc


def run_mnist_experiment(config):
    """
    Runs a PSO-CNN experiment specifically for MNIST (or simulated MNIST)
    to find a single best CNN architecture.

    Args:
        config (dict): Dictionary containing all experiment parameters.
    """
    experiment_start_time = time.time()
    logging.info("=" * 30 + " Starting MNIST Experiment " + "=" * 30)
    logging.info(f"Configuration:\n{config}")

    # --- Setup Device ---
    device = config.get('device')  # Assume device object is already created
    logging.info(f"Using device: {device}")

    # --- Load Data ---
    # Use ActionDataset configured for MNIST simulation.
    # 'early' fusion type gives a single tensor input [C, H, W], suitable for standard CNN.
    # NOTE: The channel count will be 4 due to simulation, not 1 like real MNIST.
    #       build_cnn needs to handle this input_channels value.
    logging.info("Loading dataset (MNIST simulation)...")
    try:
        train_dataset = ActionDataset(config['data_dir'], split='train', fusion_type='early')
        test_dataset = ActionDataset(config['data_dir'], split='test', fusion_type='early')

        input_channels, input_height, input_width, num_classes = train_dataset.get_details()
        logging.info(
            f"Dataset details (Simulated): Input Channels={input_channels}, H={input_height}, W={input_width}, Num Classes={num_classes}")

        # Update config with dataset details
        config['input_channels'] = input_channels  # Will be 4 due to simulation
        config['input_height'] = input_height
        config['input_width'] = input_width
        config['num_classes'] = num_classes
        config['n_out'] = num_classes  # Ensure n_out matches dataset

        # Create DataLoaders
        batch_size = config.get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=config.get('num_workers', 0))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=config.get('num_workers', 0))

    except Exception as e:
        logging.error(f"Error loading data: {e}", exc_info=True)
        return

    # --- Run PSO Architecture Search ---
    logging.info("Starting PSO architecture search for single CNN...")
    pso_config = config.copy()
    pso_config['dataset'] = train_loader  # Pass train loader for fitness evaluation

    best_particle, pso_best_loss = psoCNN(pso_config)

    if best_particle is None:
        logging.error("PSO search failed to find a valid architecture.")
        return

    logging.info(f"PSO search finished. Best particle loss during search: {pso_best_loss:.4f}")
    logging.info(f"Best architecture found (length {len(best_particle.architecture)}):")
    for l_idx, layer in enumerate(best_particle.architecture): logging.info(f"  L{l_idx + 1}: {layer}")

    # --- Final Model Training & Evaluation ---
    logging.info("Starting final training of the best architecture...")
    final_model = None
    final_train_loss = float('inf')
    test_loss = float('inf')
    test_acc = 0.0
    try:
        # Build the final model using the standard CNN builder
        final_model = build_cnn(
            architecture_encoding=best_particle.architecture,
            input_channels=config['input_channels'],  # Use the (simulated) 4 channels
            num_classes=config['num_classes'],
            config=config
        )
        final_model.to(device)

        # Set up final training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(final_model.parameters(), lr=config.get('lr', 0.001))
        epochs = config['e_test']

        logging.info(f"Training final model for {epochs} epochs...")
        # Use train_and_evaluate for simplicity, passing the TRAIN loader
        final_train_loss = train_and_evaluate(
            model=final_model,
            dataset=train_loader,  # Train on the training set
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            config=config
        )
        logging.info(f"Final model training completed. Last epoch training loss: {final_train_loss:.4f}")

        # Evaluate on the TEST set
        logging.info("Evaluating final model on test set...")
        test_loss, test_acc = evaluate_model(
            model=final_model,
            dataloader=test_loader,
            criterion=criterion,
            device=device
        )
        logging.info(f"Final Test Set Performance - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    except Exception as e:
        logging.error(f"Error during final training or evaluation: {e}", exc_info=True)

    # --- Save Results ---
    results = {
        'config': config,
        'best_particle_loss_pso': pso_best_loss,
        'best_architecture': best_particle.architecture,
        'final_model_train_loss': final_train_loss,
        'final_model_test_loss': test_loss,
        'final_model_test_accuracy': test_acc,
        'experiment_duration_seconds': time.time() - experiment_start_time,
    }
    # Make config serializable (device object isn't)
    results['config']['device'] = str(results['config']['device'])
    results_filename = f"results_mnist_{time.strftime('%Y%m%d_%H%M%S')}.json"
    results_filepath = os.path.join(config['output_dir'], results_filename)
    save_results(results, results_filepath)

    # Optional: Save final model state
    if final_model:
        model_filename = f"final_model_mnist_{time.strftime('%Y%m%d_%H%M%S')}.pth"
        model_filepath = os.path.join(config['output_dir'], model_filename)
        torch.save(final_model.state_dict(), model_filepath)
        logging.info(f"Final model state saved to {model_filepath}")

    logging.info("=" * 30 + " MNIST Experiment Finished " + "=" * 30)
    experiment_duration = time.time() - experiment_start_time
    logging.info(f"Total experiment duration: {experiment_duration:.2f} seconds")


# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PSO-CNN Architecture Search for MNIST (Simulated)")

    # --- Paths ---
    parser.add_argument('--data_dir', type=str, default="./data", help="Directory containing the 'MNIST' subfolder.")
    parser.add_argument('--output_dir', type=str, default="results_mnist",
                        help="Directory to save results, logs, and models for MNIST experiment.")

    # --- Experiment Setup ---
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use ('cuda', 'cpu', 'mps'). Auto-detects if None.")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Number of dataloader workers (set to 0 for simplicity/debugging).")
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help="Logging level.")

    # --- PSO Parameters (Defaults similar to paper Table 2 where applicable) ---
    parser.add_argument('--swarm_size', type=int, default=20, help="Number of particles (N).")
    parser.add_argument('--max_iter', type=int, default=10, help="Maximum number of PSO iterations.")  # Paper uses 10
    parser.add_argument('--max_layers', type=int, default=20,
                        help="Maximum number of functional layers (l_max).")  # Paper uses 20
    parser.add_argument('--cg', type=float, default=0.5, help="PSO parameter Cg.")  # Paper uses 0.5
    parser.add_argument('--k_max', type=int, default=7, help="Maximum Conv kernel size (odd).")  # Paper uses 7
    parser.add_argument('--maps_max', type=int, default=256, help="Maximum Conv feature maps.")  # Paper uses 256
    parser.add_argument('--n_max', type=int, default=300,
                        help="Maximum neurons in intermediate FC layers.")  # Paper uses 300

    # --- Training Parameters (Defaults similar to paper Table 2) ---
    parser.add_argument('--e_train', type=int, default=1,
                        help="Epochs for particle evaluation during PSO.")  # Paper uses 1
    parser.add_argument('--e_test', type=int, default=100, help="Epochs for final training.")  # Paper uses 100
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for Adam optimizer.")  # Common default
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size.")  # Common default for MNIST

    # --- Optional Features (Defaults similar to paper Table 2) ---
    parser.add_argument('--use_bn', action=argparse.BooleanOptionalAction, default=True,
                        help="Enable/disable Batch Normalization.")  # Paper uses BN=Yes
    parser.add_argument('--use_dropout', action=argparse.BooleanOptionalAction, default=True,
                        help="Enable/disable Dropout.")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="Dropout probability.")  # Paper uses 0.5

    args = parser.parse_args()

    # --- Setup ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    log_filename = f"log_mnist_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(args.output_dir, log_filename)
    log_level_map = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR}
    setup_logging(log_filepath, level=log_level_map.get(args.log_level, logging.INFO))

    if args.device:
        device = torch.device(args.device)
    else:
        # Auto-detect MPS or CUDA
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    config = vars(args)
    config['device'] = device

    # --- Run ---
    run_mnist_experiment(config)

