# project_root/pso/pso.py
import os  # Added for path joining and checking
import copy
import math
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import concurrent.futures
import threading
# Added imports for dataset loading in main block
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from particle import Particle  # Import the Particle class
# Assuming these relative imports work based on your project structure
from models.cnn_architecture import build_cnn
from utils.train import train_and_evaluate


# --- Helper functions for layer generation ---
def _add_random_conv(k_max, maps_max):
    """Creates a definition dictionary for a random convolutional layer."""
    k_max = max(3, k_max if k_max % 2 != 0 else k_max - 1)
    kernel_size = random.choice(list(range(3, k_max + 1, 2)))
    out_channels = random.randint(1, maps_max)
    return {'type': 'conv', 'out_channels': out_channels, 'kernel_size': kernel_size, 'stride': 1}


def _add_random_pool():
    """Creates a definition dictionary for a random pooling layer."""
    pool_type = random.choice(['max', 'avg'])
    return {'type': 'pool', 'pool_type': pool_type, 'kernel_size': 3, 'stride': 2}


def _add_random_fc(n_max):
    """Creates a definition dictionary for a random fully-connected layer."""
    neurons = random.randint(1, n_max)
    return {'type': 'fc', 'neurons': neurons}


# --- InitializeSwarm ---
def InitializeSwarm(N, l_max, maps_max, k_max, n_max, n_out,
                    use_bn=False, use_dropout=False, dropout_rate=0.5):
    """Initializes the swarm with N particles having random CNN architectures."""
    if N <= 0:
        return []
    if use_dropout and not (0 < dropout_rate < 1):
        use_dropout = False
    swarm = []
    print(
        f"Initializing swarm with {N} particles (BN: {use_bn}, Dropout: {use_dropout}, Rate: {dropout_rate if use_dropout else 'N/A'})...")
    for i in range(N):
        particle = Particle()
        actual_l_max = max(3, l_max)
        depth = random.randint(3, actual_l_max)
        list_layers = []
        functional_layers = []
        has_fc_layer_appeared = False
        # Generate functional layers first
        for j in range(depth):
            layer_index = j + 1
            if layer_index == 1:
                layer = _add_random_conv(k_max, maps_max)
            elif layer_index == depth:
                layer = {'type': 'fc', 'neurons': n_out}
                has_fc_layer_appeared = True
            elif has_fc_layer_appeared:
                layer = _add_random_fc(n_max)
            else:
                layer_type_choice = random.randint(1, 3)
                if layer_type_choice == 1:
                    layer = _add_random_conv(k_max, maps_max)
                elif layer_type_choice == 2:
                    layer = _add_random_pool()
                else:
                    layer = _add_random_fc(n_max)
                    has_fc_layer_appeared = True
            functional_layers.append(layer)
        # Add optional BN/Dropout layers
        current_has_fc = False
        for idx, layer in enumerate(functional_layers):
            is_last_functional_layer = (idx == len(functional_layers) - 1)
            list_layers.append(layer)  # Add functional layer
            if layer['type'] == 'fc':
                current_has_fc = True
            # Add BN after Conv/FC (but not last FC)
            if use_bn and layer['type'] in ('conv', 'fc') and not is_last_functional_layer:
                list_layers.append({'type': 'bn'})
            # Add Dropout after intermediate FC layers (conceptually after activation)
            if use_dropout and dropout_rate > 0 and layer['type'] == 'fc' and not is_last_functional_layer:
                list_layers.append({'type': 'dropout', 'rate': dropout_rate})
        # Assign final architecture and initialize velocity
        particle.architecture = list_layers
        particle.velocity = []  # Velocity (target arch) computed later
        swarm.append(particle)
    print(f"Successfully initialized swarm with {N} particles.")
    return swarm


# --- ComputeLoss ---
def ComputeLoss(particle, dataset, epochs, config, device, particle_id="N/A"):  # Added particle_id for logging
    """
    Computes the fitness (loss) of a particle's architecture by building,
    training (for 'epochs'), and evaluating a PyTorch model.
    Returns float('inf') on failure.
    """
    print(f"  [Particle {particle_id}] Computing loss (Arch len: {len(particle.architecture)}, Epochs: {epochs})...")
    arch_eval_start_time = time.time()
    loss = float('inf')  # Default to infinity
    try:
        input_channels = config.get('input_channels')
        num_classes = config.get('num_classes')
        learning_rate = config.get('learning_rate', 0.001)
        if input_channels is None or num_classes is None:
            raise ValueError("Config must contain 'input_channels' and 'num_classes'")

        # --- Build Model ---
        model = build_cnn(particle.architecture, input_channels, num_classes, config)
        current_device = torch.device(device)  # Ensure device is a torch.device object
        model.to(current_device)

        # --- Setup Training ---
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

        # --- Train and Evaluate ---
        computed_loss = train_and_evaluate(model=model, dataset=dataset, criterion=criterion, optimizer=optimizer,
                                           epochs=epochs, device=current_device, config=config)  # Pass device

        if computed_loss is None or math.isnan(computed_loss) or math.isinf(computed_loss):
            print(
                f"  [Particle {particle_id}] Warning: Training returned invalid loss ({computed_loss}). Setting loss to infinity.")
            loss = float('inf')
        else:
            loss = computed_loss

    except Exception as e:
        import traceback
        print(f"  [Particle {particle_id}] ERROR: Failed ComputeLoss: {e}")
        # traceback.print_exc() # Uncomment for detailed traceback
        loss = float('inf')  # Ensure loss remains infinity on error

    arch_eval_duration = time.time() - arch_eval_start_time
    # Ensure loss is not negative, clamp at 0 if needed
    loss = max(0.0, loss) if loss != float('inf') else float('inf')
    print(
        f"  [Particle {particle_id}] Loss computation finished. Result: {loss:.4f} (Duration: {arch_eval_duration:.2f}s)")
    return loss


# --- Difference Computation, Velocity Update, Particle Update ---
def _compute_difference(arch1, arch2):
    """Computes the difference (arch1 - arch2) based on layer types."""
    diff_cp = []
    diff_fc = []
    arch1_cp = [l for l in arch1 if l['type'] in ('conv', 'pool')]
    arch1_fc = [l for l in arch1 if l['type'] == 'fc']
    arch2_cp = [l for l in arch2 if l['type'] in ('conv', 'pool')]
    arch2_fc = [l for l in arch2 if l['type'] == 'fc']
    len1_cp, len2_cp = len(arch1_cp), len(arch2_cp)
    max_len_cp = max(len1_cp, len2_cp)
    for i in range(max_len_cp):
        layer1_exists = i < len1_cp
        layer2_exists = i < len2_cp
        if layer1_exists and layer2_exists:
            if arch1_cp[i]['type'] == arch2_cp[i]['type']:
                diff_cp.append(0)
            else:
                diff_cp.append(copy.deepcopy(arch1_cp[i]))
        elif layer1_exists:
            diff_cp.append(copy.deepcopy(arch1_cp[i]))
        else:
            diff_cp.append(-1)
    len1_fc, len2_fc = len(arch1_fc), len(arch2_fc)
    max_len_fc = max(len1_fc, len2_fc)
    for i in range(max_len_fc):
        layer1_exists = i < len1_fc
        layer2_exists = i < len2_fc
        if layer1_exists and layer2_exists:
            diff_fc.append(0)
        elif layer1_exists:
            diff_fc.append(copy.deepcopy(arch1_fc[i]))
        else:
            diff_fc.append(-1)
    return diff_cp, diff_fc


def UpdateParticleVelocity(particle, Cg, gBest, config):
    """Updates the particle's velocity (target architecture)."""
    if not particle.pBest_architecture:
        return copy.deepcopy(particle.architecture)
    diff_gBest_cp, diff_gBest_fc = _compute_difference(gBest.architecture, particle.architecture)
    diff_pBest_cp, diff_pBest_fc = _compute_difference(particle.pBest_architecture, particle.architecture)
    target_arch_cp = []
    target_arch_fc = []
    current_cp = [l for l in particle.architecture if l['type'] in ('conv', 'pool')]
    current_fc = [l for l in particle.architecture if l['type'] == 'fc']
    max_len_cp = max(len(diff_gBest_cp), len(diff_pBest_cp), len(current_cp))
    for i in range(max_len_cp):
        r = random.random()
        use_gBest_diff = r <= Cg
        source_diff = diff_gBest_cp if use_gBest_diff else diff_pBest_cp
        diff_element = source_diff[i] if i < len(source_diff) else None
        current_layer = current_cp[i] if i < len(current_cp) else None
        if diff_element == 0 and current_layer is not None:
            target_arch_cp.append(copy.deepcopy(current_layer))
        elif diff_element == -1:
            pass
        elif isinstance(diff_element, dict):
            target_arch_cp.append(copy.deepcopy(diff_element))
        elif diff_element is None and current_layer is not None:
            target_arch_cp.append(copy.deepcopy(current_layer))
    max_len_fc = max(len(diff_gBest_fc), len(diff_pBest_fc), len(current_fc))
    for i in range(max_len_fc):
        r = random.random()
        use_gBest_diff = r <= Cg
        source_diff = diff_gBest_fc if use_gBest_diff else diff_pBest_fc
        diff_element = source_diff[i] if i < len(source_diff) else None
        current_layer = current_fc[i] if i < len(current_fc) else None
        if diff_element == 0 and current_layer is not None:
            gBest_diff_elem = diff_gBest_fc[i] if i < len(diff_gBest_fc) else None
            pBest_diff_elem = diff_pBest_fc[i] if i < len(diff_pBest_fc) else None
            if gBest_diff_elem == 0 and pBest_diff_elem == 0:
                source_arch = gBest.architecture if use_gBest_diff else particle.pBest_architecture
                source_fc_layers = [l for l in source_arch if l['type'] == 'fc']
                if i < len(source_fc_layers):
                    target_arch_fc.append(copy.deepcopy(source_fc_layers[i]))
                else:
                    target_arch_fc.append(copy.deepcopy(current_layer))
            else:
                target_arch_fc.append(copy.deepcopy(current_layer))
        elif diff_element == -1:
            pass
        elif isinstance(diff_element, dict):
            target_arch_fc.append(copy.deepcopy(diff_element))
        elif diff_element is None and current_layer is not None:
            target_arch_fc.append(copy.deepcopy(current_layer))
    particle.velocity = target_arch_cp + target_arch_fc
    target_functional_layers = target_arch_cp + target_arch_fc

    # --- ENFORCE FINAL LAYER ---
    n_out = config.get('n_out')  # Get the required number of output neurons
    if not target_functional_layers or \
            target_functional_layers[-1].get('type') != 'fc' or \
            target_functional_layers[-1].get('neurons') != n_out:

        print(
            f"  WARNING (Velocity): Target functional layers ({len(target_functional_layers)}) didn't end correctly. Fixing.")
        # Remove any incorrect final layer(s) that might be FC but wrong size, or non-FC
        while target_functional_layers and target_functional_layers[-1].get('type') != 'fc':
            print(f"   - Removing incorrect trailing layer: {target_functional_layers[-1]}")
            target_functional_layers.pop()
        if target_functional_layers and target_functional_layers[-1].get('type') == 'fc':
            # If the last layer is FC but potentially wrong size, remove it to replace
            if target_functional_layers[-1].get('neurons') != n_out:
                print(f"   - Removing FC layer with incorrect size: {target_functional_layers[-1]}")
                target_functional_layers.pop()
            # Else: the last layer is already the correct FC layer, do nothing extra

        # If list is empty now, or the last layer isn't the correct FC layer, add it.
        if not target_functional_layers or target_functional_layers[-1].get('neurons') != n_out:
            final_fc_layer = {'type': 'fc', 'neurons': n_out}
            target_functional_layers.append(final_fc_layer)
            print(f"   + Appending required final FC layer: {final_fc_layer}")

    # The particle's velocity is set to the list of target functional layers
    particle.velocity = target_functional_layers
    # print(f"  Updated velocity (target arch functional layers: {len(particle.velocity)}).") # Debug print

    return particle.velocity  # Return the corrected list


def UpdateParticle(particle, config):
    """Updates the particle's architecture based on its velocity."""
    target_functional_layers = particle.velocity
    if not target_functional_layers:
        if particle.pBest_architecture:  # Ensure pBest exists before reverting
            particle.architecture = copy.deepcopy(particle.pBest_architecture)
        particle.velocity = []
        return particle
    new_architecture = []
    use_bn = config.get('use_bn', False)
    use_dropout = config.get('use_dropout', False)
    dropout_rate = config.get('dropout_rate', 0.5)
    n_out = config.get('n_out')
    current_has_fc = False
    for idx, layer in enumerate(target_functional_layers):
        is_last_functional_layer = (idx == len(target_functional_layers) - 1)
        new_architecture.append(layer)
        if layer['type'] == 'fc': current_has_fc = True
        if use_bn and layer['type'] in ('conv', 'fc') and not is_last_functional_layer:
            new_architecture.append({'type': 'bn'})
        if use_dropout and dropout_rate > 0 and layer['type'] == 'fc' and not is_last_functional_layer:
            new_architecture.append({'type': 'dropout', 'rate': dropout_rate})
    valid_arch = True
    if not new_architecture:
        valid_arch = False; print("  ERROR: UpdateParticle resulted in empty architecture.")
    elif new_architecture[0]['type'] != 'conv':
        valid_arch = False; print("  ERROR: UpdateParticle: First layer must be Conv.")
    elif new_architecture[-1]['type'] != 'fc' or new_architecture[-1].get('neurons') != n_out:
        valid_arch = False; print(f"  ERROR: UpdateParticle: Last layer must be FC with {n_out} neurons.")
    else:
        fc_started = False
        for layer in new_architecture:
            if layer['type'] == 'fc':
                fc_started = True
            elif fc_started and layer['type'] in ('conv', 'pool'):
                valid_arch = False; print("  ERROR: UpdateParticle: Found Conv/Pool layer after FC layer."); break
    if not valid_arch:
        print("  Warning: Invalid architecture after update. Reverting particle to pBest.")
        if particle.pBest_architecture:  # Ensure pBest exists
            particle.architecture = copy.deepcopy(particle.pBest_architecture)
        particle.velocity = []
    else:
        particle.architecture = new_architecture
    return particle


# --- Wrapper function for parallel execution ---
def evaluate_particle_task(args):
    """Helper function to unpack arguments and call ComputeLoss."""
    particle, dataset, e_train, config, device, particle_id = args
    loss = ComputeLoss(particle, dataset, e_train, config, device, particle_id)
    return particle_id, loss  # Return particle index/id and its loss


# --- Main psoCNN Function (Parallelized) ---
def psoCNN(config):
    """
    Performs Particle Swarm Optimization using multithreading for particle evaluation.
    """
    # Extract parameters...
    try:
        N = config['N']
        iter_max = config['iter_max']
        dataset = config['dataset']  # DataLoader
        l_max = config['l_max']
        Cg = config['Cg']
        k_max = config['k_max']
        maps_max = config['maps_max']
        n_max = config['n_max']
        n_out = config['n_out']
        e_train = config['e_train']
        e_test = config['e_test']
        device = config['device']  # Should be torch.device object
        use_bn = config.get('use_bn', False)
        use_dropout = config.get('use_dropout', False)
        dropout_rate = config.get('dropout_rate', 0.5)
        max_workers = config.get('max_workers', 4)

        # Basic config validation (replace [...] with actual checks)
        # if not all([...]):
        #    raise ValueError("Invalid configuration parameter types or values.")

    except KeyError as e:
        print(f"ERROR: Missing required config key: {e}"); return None, float('inf')
    except ValueError as e:
        print(f"ERROR: Invalid config value: {e}"); return None, float('inf')

    print("--- Starting psoCNN (Parallel Version) ---")
    print(f"Config: N={N}, iter={iter_max}, ..., max_workers={max_workers}, device={device}")

    # 1. Initialize Swarm (Sequential)
    swarm = InitializeSwarm(N, l_max, maps_max, k_max, n_max, n_out, use_bn, use_dropout, dropout_rate)
    if not swarm: print("ERROR: Swarm initialization failed."); return None, float('inf')

    # 2. Initialize Global Best and Lock
    gBest = None
    gBest_loss = float('inf')
    gBest_lock = threading.Lock()  # Lock for protecting gBest updates

    # 3. Initial Evaluation (Parallel)
    print(f"--- Initial Evaluation Phase (Parallel with {max_workers} workers) ---")
    initial_eval_start_time = time.time()
    tasks_args = [(swarm[i], dataset, e_train, config, device, i) for i in range(N)]
    particle_losses = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_particle_task, arg): i for i, arg in enumerate(tasks_args)}
        for future in concurrent.futures.as_completed(futures):
            particle_id = futures[future]
            try:
                p_id, loss = future.result()  # p_id should match particle_id
                particle_losses[p_id] = loss
                swarm[p_id].loss = loss
                # Set initial pBest regardless of loss value
                swarm[p_id].pBest_architecture = copy.deepcopy(swarm[p_id].architecture)
                swarm[p_id].pBest_loss = loss

                # --- Thread-Safe gBest Update ---
                with gBest_lock:
                    if loss < gBest_loss:
                        gBest_loss = loss
                        gBest = copy.deepcopy(swarm[p_id])
                        print(f"  Initial gBest updated by Particle {p_id}! Loss: {gBest_loss:.4f}")
            except Exception as exc:
                print(f'  Particle {particle_id} generated an exception during initial eval: {exc}')
                particle_losses[particle_id] = float('inf')
                swarm[particle_id].loss = float('inf')
                # Still set initial pBest to current state, even if loss is Inf
                swarm[particle_id].pBest_architecture = copy.deepcopy(swarm[particle_id].architecture)
                swarm[particle_id].pBest_loss = float('inf')

    initial_eval_duration = time.time() - initial_eval_start_time
    print(f"--- Initial Evaluation Duration: {initial_eval_duration:.2f}s ---")

    if gBest is None:
        print("ERROR: Initial evaluation failed for all particles or resulted in Inf loss.")
        return None, float('inf')
    print(f"--- Initial gBest Loss: {gBest.loss:.4f} ---")

    # --- 4. Main PSO loop ---
    print(f"--- Starting PSO Iterations (Parallel with {max_workers} workers) ---")
    iteration_start_time = time.time()

    for iter_num in range(iter_max):
        iter_loop_start_time = time.time()
        print(f"\n--- Iteration {iter_num + 1}/{iter_max} ---")
        current_iter_gBest_loss = gBest_loss  # Track improvement

        # --- Update Velocities and Architectures (Sequential) ---
        print("  Updating particle velocities and architectures (sequentially)...")
        for i, Pi in enumerate(swarm):
            if gBest:
                Pi.velocity = UpdateParticleVelocity(Pi, Cg, gBest, config)
                Pi = UpdateParticle(Pi, config)  # Updates Pi.architecture
                swarm[i] = Pi
            else:
                print(f"  Warning: Skipping velocity/architecture update for particle {i} as gBest is not yet valid.")

        # --- Evaluate Updated Particles (Parallel) ---
        print(f"  Evaluating updated particles (parallel with {max_workers} workers)...")
        eval_tasks_args = [(swarm[i], dataset, e_train, config, device, i) for i in range(N)]
        iteration_losses = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_particle_task, arg): i for i, arg in enumerate(eval_tasks_args)}
            for future in concurrent.futures.as_completed(futures):
                particle_id = futures[future]
                try:
                    p_id, loss = future.result()
                    iteration_losses[p_id] = loss
                    swarm[p_id].loss = loss  # Update particle's current loss

                    # --- Update pBest ---
                    if loss < swarm[p_id].pBest_loss:
                        print(f"  Particle {p_id}: New pBest! Loss: {loss:.4f} (was {swarm[p_id].pBest_loss:.4f})")
                        swarm[p_id].pBest_architecture = copy.deepcopy(swarm[p_id].architecture)
                        swarm[p_id].pBest_loss = loss

                        # --- Thread-Safe gBest Update ---
                        with gBest_lock:
                            if swarm[p_id].pBest_loss < gBest_loss:
                                gBest_loss = swarm[p_id].pBest_loss
                                gBest = copy.deepcopy(swarm[p_id])  # Deep copy the best particle
                                print(f"  GLOBAL BEST updated by Particle {p_id}! Loss: {gBest_loss:.4f}")

                except Exception as exc:
                    print(f'  Particle {particle_id} generated an exception during iteration eval: {exc}')
                    iteration_losses[particle_id] = float('inf')
                    swarm[particle_id].loss = float('inf')  # Update current loss to inf

        iter_loop_duration = time.time() - iter_loop_start_time
        print(
            f"--- End Iteration {iter_num + 1} --- gBest Loss: {gBest_loss:.4f} --- Duration: {iter_loop_duration:.2f}s ---")
        if gBest_loss >= current_iter_gBest_loss:
            print("  (gBest loss did not improve this iteration)")

    total_iteration_duration = time.time() - iteration_start_time
    print(f"\n--- PSO Iterations Complete (Total Duration: {total_iteration_duration:.2f}s) ---")

    # --- 5. Final Training/Evaluation (Sequential) ---
    print("\n--- Final Training/Evaluation of Best Architecture ---")
    if gBest is None:
        print("ERROR: No valid gBest particle found after iterations.")
        return None, float('inf')

    print(f"Best architecture found (total layers: {len(gBest.architecture)}):")
    # Log the best architecture found
    functional_layers_count = len([l for l in gBest.architecture if l['type'] in ('conv', 'pool', 'fc')])
    print(f"(Functional layers: {functional_layers_count})")
    # Optional: Print details
    # for l_idx, layer in enumerate(gBest.architecture): print(f"  L{l_idx+1}: {layer}")

    print(f"Starting final training for {e_test} epochs...")
    # ComputeLoss can be reused for final training
    final_loss = ComputeLoss(gBest, dataset, e_test, config, device, particle_id="gBest-Final")
    gBest.loss = final_loss  # Update final loss

    print(f"--- Final Loss after {e_test} epochs: {gBest.loss:.4f} ---")

    # 6. Return the best particle and its final loss
    return gBest, gBest.loss


# --- Example Usage (Standalone Test - MODIFIED FOR CIFAR-10) ---
if __name__ == '__main__':

    print("\n" + "=" * 30 + "\n--- Running Standalone pso.py Test (Parallel, CIFAR-10) ---" + "\n" + "=" * 30)

    # --- CIFAR-10 Data Loading ---
    # Use '../data' assuming script is run from thesis_code/pso/
    data_dir = '../data'
    if not os.path.exists(data_dir):
        # Attempt to create if running from pso dir, might fail depending on permissions
        try:
            os.makedirs(data_dir)
            print(f"Created data directory: {data_dir}")
        except OSError as e:
            print(f"Could not create data directory {data_dir}. Please ensure it exists. Error: {e}")
            exit()  # Exit if data directory cannot be created/found

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        # Load only train set for particle evaluation in this test
        cifar_train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        # Use a small batch size for testing, set num_workers=0 for simplicity with threading
        # Use only a subset for quick testing if desired (e.g., SubsetRandomSampler)
        train_loader = DataLoader(cifar_train_dataset, batch_size=32, shuffle=True, num_workers=0)
        print(f"CIFAR-10 training data loaded successfully from '{data_dir}'.")
    except Exception as e:
        print(f"Error loading CIFAR-10 dataset from {data_dir}: {e}")
        print("Ensure the path is correct relative to where you run pso.py and you have permissions/connectivity.")
        exit()  # Exit if data cannot be loaded

    # Example Configuration (Updated for CIFAR-10)
    test_config = {
        'N': 4,  # Small swarm for testing
        'iter_max': 2,  # Few iterations for testing
        'dataset': train_loader,  # Use the loaded CIFAR-10 DataLoader
        'l_max': 12,  # Max functional layers
        'Cg': 0.5, # Cognitive factor
        'k_max': 5,  # Max kernel size
        'maps_max': 64,  # Max feature maps
        'n_max': 128,  # Max FC neurons
        'n_out': 10,  # CIFAR-10 has 10 classes
        'e_train': 2,  # Quick epoch for particle eval
        'e_test': 6,  # Quick epochs for final eval test
        'input_channels': 3,  # CIFAR-10 has 3 color channels
        'num_classes': 10,  # CIFAR-10 has 10 classes
        'learning_rate': 0.001,
        'device': torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
        'use_bn': True,
        'use_dropout': True,
        'dropout_rate': 0.25,
        'max_workers': 3  # Set number of parallel workers
        # 'num_workers': 0 # Explicitly setting dataloader workers to 0 in config if needed elsewhere
    }
    print(f"\nTest Configuration:\n{test_config}\n")

    # --- Run PSO-CNN ---
    overall_start_time = time.time()
    best_particle_found, final_best_loss = psoCNN(test_config)
    overall_duration = time.time() - overall_start_time

    # --- Results ---
    print("\n" + "=" * 30 + "\n--- Standalone pso.py Test Finished ---" + "\n" + "=" * 30)
    print(f"Overall Execution Time: {overall_duration:.2f}s")
    if best_particle_found:
        print(f"Best Particle Final Loss: {final_best_loss:.4f}")
        print("Best Architecture Found:")
        for i, layer in enumerate(best_particle_found.architecture):
            print(f"  Layer {i + 1}: {layer}")
    else:
        print("PSO Run Failed or No Valid Architecture Found.")
    print("=" * 30)