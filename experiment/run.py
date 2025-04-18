import os
import logging
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- Project Imports ---
# Assume these modules exist and are importable
try:
    from data_preprocessing.dataset import ActionDataset  # Needs implementation!
    from models.fusion_models import build_early_fusion_model, build_late_fusion_model  # Needs implementation!
    from pso.pso import psoCNN
    from utils.helpers import save_results, save_checkpoint, load_checkpoint  # Assumes helpers.py exists
    from utils.train import train_and_evaluate, evaluate_model  # Assumes train.py exists
except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}. Ensure all required files exist.")


    # Define dummy placeholders if imports fail, to allow script structure check
    class ActionDataset:  # Dummy Dataset
        def __init__(self, data_dir, split='train', fusion_type='early'):
            self.len = 100; self.fusion_type = fusion_type
            self.channels = 8 if fusion_type == 'early' else (6, 2)
            self.h = 32; self.w = 32; self.classes = 10

        def __len__(self):
            return self.len

        def __getitem__(self, idx):
            if self.fusion_type == 'early':
                return torch.randn(self.channels, self.h, self.w), idx % self.classes
            else:
                return torch.randn(self.channels[0], self.h, self.w), torch.randn(self.channels[1], self.h,
                                                                                  self.w), idx % self.classes

        def get_details(self):
            return self.channels, self.h, self.w, self.classes


    def build_early_fusion_model(arch, in_c, n_cls, cfg):
        logging.info("DUMMY: Build Early Fusion"); return nn.Linear(10, n_cls)  # Minimal dummy


    def build_late_fusion_model(arch_c, arch_d, in_c, in_d, n_cls, cfg):
        logging.info("DUMMY: Build Late Fusion"); return nn.Linear(10, n_cls)  # Minimal dummy


    def psoCNN(cfg):
        logging.warning("DUMMY: psoCNN called")
        return type('obj', (object,), {'architecture': [{'type': 'fc', 'neurons': cfg['n_out']}]})(), 0.5  # Dummy best particle + loss


    def save_results(res, fp):
        logging.info(f"DUMMY: Save results to {fp}")


    def save_checkpoint(m, o, e, l, fp):
        logging.info(f"DUMMY: Save checkpoint to {fp}")


    def train_and_evaluate(model, dataset, criterion, optimizer, epochs, device, config):
        logging.warning("DUMMY: train_and_evaluate called"); return 0.5  # Dummy loss


    def evaluate_model(model, dataset, criterion, device):
        logging.warning("DUMMY: evaluate_model called"); return 0.4, 0.9  # Dummy loss, acc


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
        # val_dataset = ActionDataset(config['data_dir'], split='val', fusion_type=config['fusion_type']) # Optional
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
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.get('num_workers', 2))
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

        # Train the model fully (using train_and_evaluate for simplicity here)
        # A more robust training loop might involve validation and saving best checkpoint
        logging.info(f"Training final model for {epochs} epochs...")
        # Use train_and_evaluate for consistency with ComputeLoss evaluation method
        # Pass the TRAIN loader
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