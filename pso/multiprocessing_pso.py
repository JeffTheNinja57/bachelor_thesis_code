import copy
import time
import multiprocessing as mp

import torch

from .initialize_swarm import InitializeSwarm
from .pso_helpers import ComputeLoss
from .pso_helpers import UpdateParticle
from .pso_helpers import UpdateParticleVelocity


def evaluate_particle(particle, dataset, e_train, config, device_str, result_queue, particle_idx):
    """
    Worker function to evaluate a particle in a separate process.

    Args:
        particle: The particle to evaluate
        dataset: The dataset for training
        e_train: Number of epochs for training
        config: Configuration dictionary
        device_str: Device string (converted to torch.device in the worker)
        result_queue: Queue to store results
        particle_idx: Index of the particle in the swarm
    """
    try:
        # Convert device string to torch.device
        device = torch.device(device_str)

        # Clear GPU cache if using MPS
        if device.type == 'mps':
            torch.mps.empty_cache()

        print(f"  Process {mp.current_process().name}: Evaluating particle {particle_idx}...")

        # Compute loss for the particle
        loss = ComputeLoss(particle, dataset, e_train, config, device)

        # Update particle's loss and pBest
        particle.loss = loss
        if loss < particle.pBest_loss:
            particle.pBest_architecture = copy.deepcopy(particle.architecture)
            particle.pBest_loss = loss

        # Ensure any tensors in the particle are moved to CPU before sharing
        # This is a precaution to avoid "_share_filename_: only available on CPU" errors
        # when sharing tensors between processes
        if hasattr(particle, 'model') and particle.model is not None:
            particle.model = particle.model.cpu()

        # Put result in queue
        result_queue.put((particle_idx, particle, loss))

    except Exception as e:
        print(f"  ERROR in process {mp.current_process().name}: {e}")
        # Return a failed result
        result_queue.put((particle_idx, None, float('inf')))


def multiprocessing_psoCNN(config):
    """
    Performs Particle Swarm Optimization to find a good CNN architecture using multiprocessing.

    This version evaluates multiple particles in parallel using separate processes.

    Args:
        config (dict): A dictionary containing all necessary parameters:
            N, iter_max, dataset, l_max, Cg, k_max, n_max, n_out,
            e_train, e_test, maps_max, input_channels, num_classes,
            learning_rate, device, use_bn, use_dropout, dropout_rate, etc.
            num_processes (optional): Number of processes to use for parallelization.
                                     If not provided, uses max(1, cpu_count() - 1)

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

        # Determine number of processes to use
        num_processes = config.get('num_processes', max(1, mp.cpu_count() - 1))

        # Convert device to string for passing to worker processes
        device_str = str(device)

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

    print("--- Starting multiprocessing_psoCNN ---")
    print(
        f"Config: N={N}, iter={iter_max}, l_max={l_max}, max_fc_layers={max_fc_layers}, Cg={Cg}, k_max={k_max}, maps_max={maps_max}, n_max={n_max}, n_out={n_out}, e_train={e_train}, e_test={e_test}, device={device}, BN={use_bn}, Dropout={use_dropout}")
    print(f"Using {num_processes} processes for parallel evaluation")

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

    # Create a pool of workers for initial evaluation
    result_queue = mp.Queue()
    processes = []

    # Start processes for initial evaluation
    for i, Pi in enumerate(swarm):
        p = mp.Process(
            target=evaluate_particle,
            args=(Pi, dataset, e_train, config, device_str, result_queue, i)
        )
        processes.append(p)
        p.start()
        # Stagger starts to avoid memory spikes
        time.sleep(1)

    # Collect results and update swarm
    for _ in range(N):
        idx, particle, loss = result_queue.get()
        if particle is not None:
            swarm[idx] = particle

            # Update global best if needed
            if loss < gBest_loss:
                gBest = copy.deepcopy(particle)
                gBest_loss = loss
                print(f"  New initial gBest found! Particle {idx + 1}, Loss: {gBest_loss:.4f}")

    # Wait for all processes to finish
    for p in processes:
        p.join()

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

        # Update velocities for all particles
        for i, Pi in enumerate(swarm):
            Pi.velocity = UpdateParticleVelocity(Pi, Cg, gBest)
            Pi = UpdateParticle(Pi, config)  # Update particle in place
            swarm[i] = Pi

        # Evaluate particles in parallel
        result_queue = mp.Queue()
        processes = []

        # Start processes for evaluation
        for i, Pi in enumerate(swarm):
            p = mp.Process(
                target=evaluate_particle,
                args=(Pi, dataset, e_train, config, device_str, result_queue, i)
            )
            processes.append(p)
            p.start()
            # Stagger starts to avoid memory spikes
            time.sleep(1)

        # Collect results and update swarm
        for _ in range(N):
            idx, particle, loss = result_queue.get()
            if particle is not None:
                swarm[idx] = particle

                # Update global best if needed
                if particle.pBest_loss < gBest_loss:
                    print(f"  Particle {idx + 1}: New gBest found! Loss: {particle.pBest_loss:.4f} (was {gBest_loss:.4f})")
                    gBest = copy.deepcopy(particle)
                    gBest_loss = particle.pBest_loss

        # Wait for all processes to finish
        for p in processes:
            p.join()

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


# --- Example Usage (for testing multiprocessing_pso.py standalone) ---
if __name__ == '__main__':

    print("\n" + "=" * 30 + "\n--- Running Standalone multiprocessing_pso.py Test ---" + "\n" + "=" * 30)

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
        'num_processes': 3,  # Number of processes to use
    }
    print(f"Test Configuration:\n{test_config}\n")

    # Run PSO-CNN
    overall_start_time = time.time()
    best_particle_found, final_best_loss = multiprocessing_psoCNN(test_config)
    overall_duration = time.time() - overall_start_time

    print("\n" + "=" * 30 + "\n--- Standalone multiprocessing_pso.py Test Finished ---" + "\n" + "=" * 30)
    if best_particle_found:
        print(f"Overall Execution Time: {overall_duration:.2f}s")
        print(f"Best Particle Final Loss: {final_best_loss:.4f}")
        print("Best Architecture Found:")
        for i, layer in enumerate(best_particle_found.architecture):
            print(f"  Layer {i + 1}: {layer}")
    else:
        print("PSO Run Failed or No Valid Architecture Found.")
    print("=" * 30)
