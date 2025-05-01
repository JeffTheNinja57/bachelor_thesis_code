import copy
import time

import torch

from .initialize_swarm import InitializeSwarm
from .pso_helpers import ComputeLoss
from .pso_helpers import UpdateParticle
from .pso_helpers import UpdateParticleVelocity


def psoCNN(config):
    """
    Performs Particle Swarm Optimization to find a good CNN architecture.

    Args:
        config (dict): A dictionary containing all necessary parameters:
            N, iter_max, dataset, l_max, Cg, k_max, n_max, n_out,
            e_train, e_test, maps_max, input_channels, num_classes,
            learning_rate, device, use_bn, use_dropout, dropout_rate, etc.

    Returns:
        tuple: (best_particle, best_loss)
               best_particle (Particle): The particle with the best architecture found.
               best_loss (float): The loss of the best architecture after final training.
               Returns (None, float('inf')) on failure.
    """
    # Extract parameters from config
    try:
        N = config['N']
        iter_max = config['iter_max']
        dataset = config['dataset']  # Assume dataset is loaded and ready (e.g., DataLoader)
        l_max = config['l_max']
        Cg = config['Cg']
        k_max = config['k_max']
        maps_max = config['maps_max']
        n_max = config['n_max']
        n_out = config['n_out']
        e_train = config['e_train']
        e_test = config['e_test']
        device = config['device']
        use_bn = config.get('use_bn', False)  # Optional params
        use_dropout = config.get('use_dropout', False)
        dropout_rate = config.get('dropout_rate', 0.5)
        max_fc_layers = config.get('max_fc_layers', 5)  # Default to 5 if not specified

        # Basic validation
        if not all([isinstance(N, int), N > 0, isinstance(iter_max, int), iter_max >= 0,
                    isinstance(l_max, int), l_max >= 3, isinstance(Cg, float), 0 <= Cg <= 1,
                    isinstance(k_max, int), k_max >= 3, isinstance(maps_max, int), maps_max > 0,
                    isinstance(n_max, int), n_max > 0, isinstance(n_out, int), n_out > 0,
                    isinstance(e_train, int), e_train > 0, isinstance(e_test, int), e_test > 0]):
            raise ValueError("Invalid configuration parameter types or values.")

    except KeyError as e:
        print(f"ERROR: Missing required configuration key: {e}")
        return None, float('inf')
    except ValueError as e:
        print(f"ERROR: Invalid configuration value: {e}")
        return None, float('inf')

    print("--- Starting psoCNN ---")
    print(
        f"Config: N={N}, iter={iter_max}, l_max={l_max}, max_fc_layers={max_fc_layers}, Cg={Cg}, k_max={k_max}, maps_max={maps_max}, n_max={n_max}, n_out={n_out}, e_train={e_train}, e_test={e_test}, device={device}, BN={use_bn}, Dropout={use_dropout}")

    # 1. Initialize the swarm
    swarm = InitializeSwarm(N, l_max, maps_max, k_max, n_max, n_out,
                            use_bn, use_dropout, dropout_rate, max_fc_layers)
    if not swarm:
        print("ERROR: Swarm initialization failed.")
        return None, float('inf')

    # 2. Initialize personal and global bests
    print("--- Initial Evaluation Phase ---")
    gBest = None
    gBest_loss = float('inf')

    for i, Pi in enumerate(swarm):
        print(f"Evaluating initial particle {i + 1}/{N}...")
        Pi.loss = ComputeLoss(Pi, dataset, e_train, config, device)
        # Set initial pBest regardless of loss value (it's the best seen *so far* for this particle)
        Pi.pBest_architecture = copy.deepcopy(Pi.architecture)
        Pi.pBest_loss = Pi.loss

        if Pi.loss < gBest_loss:  # Use < to avoid replacing gBest with identical loss initially
            gBest = copy.deepcopy(Pi)  # Store a copy
            gBest_loss = Pi.loss
            print(f"  New initial gBest found! Particle {i + 1}, Loss: {gBest_loss:.4f}")

    if gBest is None:
        print("ERROR: Initial evaluation failed for all particles or resulted in Inf loss.")
        return None, float('inf')

    print(f"--- Initial gBest Loss: {gBest.loss:.4f} ---")

    # --- 4. Main PSO loop ---
    print("--- Starting PSO Iterations ---")
    iteration_start_time = time.time()
    for iter_num in range(iter_max):
        iter_loop_start_time = time.time()
        print(f"\n--- Iteration {iter_num + 1}/{iter_max} ---")
        current_iter_gBest_loss = gBest_loss  # Track if gBest improves this iteration

        for i, Pi in enumerate(swarm):
            particle_start_time = time.time()
            print(f"Processing Particle {i + 1}/{N} (Current Loss: {Pi.loss:.4f}, pBest: {Pi.pBest_loss:.4f})")

            # Update velocity (target architecture)
            Pi.velocity = UpdateParticleVelocity(Pi, Cg, gBest)

            # Update particle architecture based on velocity
            Pi = UpdateParticle(Pi, config)  # Update particle in place

            # Compute new loss only if architecture potentially changed
            # (Could add check: if Pi.architecture != Pi.pBest_architecture or first iter)
            print(f"  Evaluating updated particle {i + 1}...")
            Pi.loss = ComputeLoss(Pi, dataset, e_train, config, device)

            # Update personal best
            if Pi.loss < Pi.pBest_loss:  # Strictly better
                print(f"  Particle {i + 1}: New pBest found! Loss: {Pi.loss:.4f} (was {Pi.pBest_loss:.4f})")
                Pi.pBest_architecture = copy.deepcopy(Pi.architecture)
                Pi.pBest_loss = Pi.loss

                # Update global best if this new pBest is better
                if Pi.pBest_loss < gBest_loss:  # Strictly better
                    print(f"  Particle {i + 1}: New gBest found! Loss: {Pi.pBest_loss:.4f} (was {gBest_loss:.4f})")
                    gBest = copy.deepcopy(Pi)  # Store a copy
                    gBest_loss = Pi.pBest_loss

            # Store updated particle back into swarm (already updated in place)
            swarm[i] = Pi
            particle_duration = time.time() - particle_start_time
            print(f"  Particle {i + 1} processing time: {particle_duration:.2f}s")

        iter_loop_duration = time.time() - iter_loop_start_time
        print(
            f"--- End Iteration {iter_num + 1} --- gBest Loss: {gBest_loss:.4f} --- Duration: {iter_loop_duration:.2f}s ---")
        if gBest_loss >= current_iter_gBest_loss:
            print("  (gBest loss did not improve this iteration)")

    total_iteration_duration = time.time() - iteration_start_time
    print(f"\n--- PSO Iterations Complete (Total Duration: {total_iteration_duration:.2f}s) ---")

    # --- 5. Retrain/Evaluate the best architecture found ---
    print("\n--- Final Training/Evaluation of Best Architecture ---")
    if gBest is None:
        print("ERROR: No valid gBest particle found after iterations.")
        return None, float('inf')

    print(
        f"Best architecture found (functional layers: {len([l for l in gBest.architecture if l['type'] in ('conv', 'pool', 'fc')])}, total: {len(gBest.architecture)}):")
    # Optional: Print the best architecture layers
    # for l_idx, layer in enumerate(gBest.architecture): print(f"  L{l_idx+1}: {layer}")

    # Re-evaluate loss using e_test epochs for the final performance measure
    print(f"Starting final training for {e_test} epochs...")
    final_loss = ComputeLoss(gBest, dataset, e_test, config, device)
    gBest.loss = final_loss  # Update the gBest particle's loss with the final value

    print(f"--- Final Loss after {e_test} epochs: {gBest.loss:.4f} ---")

    # Ensure any models in the particle are moved to CPU before returning
    # This is crucial to avoid "_share_filename_: only available on CPU" errors
    # when the particle is shared between processes
    if hasattr(gBest, 'model') and gBest.model is not None:
        gBest.model = gBest.model.cpu()

    # 6. Return the best particle and its final loss
    return gBest, gBest.loss


# --- Example Usage (for testing pso.py standalone) ---
if __name__ == '__main__':

    print("\n" + "=" * 30 + "\n--- Running Standalone pso.py Test ---" + "\n" + "=" * 30)

    # Dummy dataset placeholder
    # In a real run, this would be PyTorch DataLoader(s) for train/val splits
    dummy_dataset_placeholder = "dummy_data_loader"

    # Example Configuration
    test_config = {
        'N': 6,  # Swarm size
        'iter_max': 4,  # Max iterations
        'dataset': dummy_dataset_placeholder,  # Use placeholder
        'l_max': 7,  # Max functional layers
        'max_fc_layers': 5,  # Maximum number of fully connected layers
        'Cg': 0.7,  # gBest probability factor
        'k_max': 5,  # Max kernel size (e.g., 5x5)
        'maps_max': 16,  # Max feature maps per conv layer
        'n_max': 32,  # Max neurons per intermediate FC layer
        'n_out': 5,  # Number of output classes
        'e_train': 2,  # Epochs for particle evaluation (quick)
        'e_test': 5,  # Epochs for final training
        'input_channels': 3,  # Example for RGB images
        'num_classes': 5,  # Matches n_out
        'learning_rate': 0.005,  # Learning rate for Adam
        'device': torch.device("mps" if torch.mps.is_available() else "cpu"),  # Set device
        'use_bn': True,  # Enable Batch Normalization
        'use_dropout': True,  # Enable Dropout
        'dropout_rate': 0.3,  # Dropout probability
    }
    print(f"Test Configuration:\n{test_config}\n")

    # Run PSO-CNN
    overall_start_time = time.time()
    best_particle_found, final_best_loss = psoCNN(test_config)
    overall_duration = time.time() - overall_start_time

    print("\n" + "=" * 30 + "\n--- Standalone pso.py Test Finished ---" + "\n" + "=" * 30)
    if best_particle_found:
        print(f"Overall Execution Time: {overall_duration:.2f}s")
        print(f"Best Particle Final Loss: {final_best_loss:.4f}")
        print("Best Architecture Found:")
        for i, layer in enumerate(best_particle_found.architecture):
            print(f"  Layer {i + 1}: {layer}")
    else:
        print("PSO Run Failed or No Valid Architecture Found.")
    print("=" * 30)
