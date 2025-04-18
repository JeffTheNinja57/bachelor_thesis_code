import argparse
import os
import torch
import logging
import json
import time

try:
    from experiment.run import run_pso_experiment
    from utils.helpers import setup_logging
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}. Ensure project structure is correct.")


    # Define dummy placeholders if imports fail
    def run_pso_experiment(config):
        print(f"DUMMY: Running experiment with config: {config}")


    def setup_logging(log_file, level):
        print(f"DUMMY: Setup logging to {log_file}")


def main():
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

    # --- Training Parameters ---
    parser.add_argument('--e_train', type=int, default=5, help="Epochs for particle evaluation during PSO.")
    parser.add_argument('--e_test', type=int, default=50, help="Epochs for final training of the best model.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for Adam optimizer.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and evaluation.")

    # --- Optional Features ---
    parser.add_argument('--use_bn', action='store_true', help="Enable Batch Normalization in architectures.")
    parser.add_argument('--use_dropout', action='store_true', help="Enable Dropout in architectures.")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="Dropout probability if --use_dropout is set.")

    args = parser.parse_args()

    # --- Post-processing and Setup ---
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Setup Logging
    log_filename = f"log_{args.fusion_type}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(args.output_dir, log_filename)
    log_level_map = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR}
    setup_logging(log_filepath, level=log_level_map.get(args.log_level, logging.INFO))

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.mps.is_available() else "cpu")
    logging.info(f"Selected device: {device}")

    # Convert args to config dictionary
    config = vars(args)  # Converts Namespace to dict
    config['device'] = device
    # Add any other fixed configurations if needed
    # config['some_other_param'] = value

    logging.info("Configuration prepared. Starting experiment...")
    # --- Run Experiment ---
    try:
        run_pso_experiment(config)
        logging.info("Experiment finished successfully.")
    except Exception as e:
        logging.exception(f"An error occurred during the experiment: {e}")  # Log traceback


if __name__ == '__main__':
    main()
