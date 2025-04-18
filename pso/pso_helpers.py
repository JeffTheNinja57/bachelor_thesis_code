import copy
import math
import random
import time

import torch.nn as nn
import torch.optim as optim

from .particle import Particle
from models.cnn_architecture import build_cnn
from utils.train import train_and_evaluate


def ComputeLoss(particle, dataset, epochs, config, device):
    """
    Computes the fitness (loss) of a particle's architecture by building,
    training (for 'epochs'), and evaluating a PyTorch model.
    Returns float('inf') on failure.
    """
    print(f"  Computing loss for particle (Arch len: {len(particle.architecture)}, Epochs: {epochs})...")
    arch_eval_start_time = time.time()
    try:
        input_channels = config.get('input_channels')
        num_classes = config.get('num_classes')
        learning_rate = config.get('learning_rate', 0.001)
        if input_channels is None or num_classes is None:
            raise ValueError("Config must contain 'input_channels' and 'num_classes'")

        model = build_cnn(particle.architecture, input_channels, num_classes, config)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        # Filter params requiring grad - important if parts of model are frozen (not typical here)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

        loss = train_and_evaluate(model=model, dataset=dataset, criterion=criterion, optimizer=optimizer,
                                  epochs=epochs, device=device, config=config)

        if loss is None or math.isnan(loss) or math.isinf(loss):
            print("  Warning: Training returned invalid loss (None, NaN or Inf). Setting loss to infinity.")
            loss = float('inf')
    except Exception as e:
        import traceback
        print(f"  ERROR: Failed ComputeLoss for particle: {e}")
        # traceback.print_exc() # Uncomment for detailed traceback during debugging
        # print(f"  Architecture causing error: {particle.architecture}") # Optional
        loss = float('inf')

    arch_eval_duration = time.time() - arch_eval_start_time
    # Ensure loss is not negative, clamp at 0 if needed (though unlikely with CrossEntropy)
    loss = max(0.0, loss) if loss != float('inf') else float('inf')
    print(f"  Loss computation finished. Result: {loss:.4f} (Duration: {arch_eval_duration:.2f}s)")
    return loss


# --- PSO Helper: Difference Computation ---

def _compute_difference(arch1, arch2):
    """
    Computes the difference (arch1 - arch2) based on layer types, separating
    Conv/Pool (CP) and Fully-Connected (FC) sections.

    Args:
        arch1 (list): Architecture encoding (list of layer dicts).
        arch2 (list): Architecture encoding (list of layer dicts).

    Returns:
        tuple[list, list]: A tuple containing two difference lists:
                           (diff_cp, diff_fc). Elements are 0, -1, or layer dict (+L).
    """
    diff_cp = []
    diff_fc = []

    # Separate layers, excluding BN/Dropout for difference calculation
    arch1_cp = [l for l in arch1 if l['type'] in ('conv', 'pool')]
    arch1_fc = [l for l in arch1 if l['type'] == 'fc']
    arch2_cp = [l for l in arch2 if l['type'] in ('conv', 'pool')]
    arch2_fc = [l for l in arch2 if l['type'] == 'fc']

    # Compare Conv/Pool sections
    len1_cp, len2_cp = len(arch1_cp), len(arch2_cp)
    max_len_cp = max(len1_cp, len2_cp)
    for i in range(max_len_cp):
        layer1_exists = i < len1_cp
        layer2_exists = i < len2_cp
        if layer1_exists and layer2_exists:
            # Layer exists in both, compare type
            if arch1_cp[i]['type'] == arch2_cp[i]['type']:
                diff_cp.append(0)  # Same type
            else:
                # Different types -> target should be layer from arch1
                diff_cp.append(copy.deepcopy(arch1_cp[i]))  # Add layer L from arch1
        elif layer1_exists:  # Only in arch1 -> target should add it
            diff_cp.append(copy.deepcopy(arch1_cp[i]))  # Add layer L from arch1
        else:  # Only in arch2 -> target should remove it
            diff_cp.append(-1)  # Remove layer

    # Compare FC sections
    len1_fc, len2_fc = len(arch1_fc), len(arch2_fc)
    max_len_fc = max(len1_fc, len2_fc)
    for i in range(max_len_fc):
        layer1_exists = i < len1_fc
        layer2_exists = i < len2_fc
        if layer1_exists and layer2_exists:
            # Both have FC layer, type matches -> 0 difference based on type
            diff_fc.append(0)
        elif layer1_exists:  # Only in arch1 -> target should add it
            diff_fc.append(copy.deepcopy(arch1_fc[i]))  # Add layer L from arch1
        else:  # Only in arch2 -> target should remove it
            diff_fc.append(-1)  # Remove layer

    return diff_cp, diff_fc


# --- PSO Helper: Velocity Update ---

def UpdateParticleVelocity(particle, Cg, gBest):
    """
    Updates the particle's velocity, which represents the target architecture
    for the next step, based on differences towards pBest and gBest.

    Args:
        particle (Particle): The particle to update.
        Cg (float): Probability factor [0, 1] for choosing gBest influence.
        gBest (Particle): The global best particle in the swarm.

    Returns:
        list: The newly computed velocity vector (target architecture layers).
              Returns current architecture if pBest is empty (initial state).
    """
    if not particle.pBest_architecture:  # Handle initial state before first pBest update
        print("  Warning: pBest not set for particle, cannot compute velocity. Keeping current arch.")
        return copy.deepcopy(particle.architecture)

    # Compute differences relative to the current particle's architecture
    # Note: Differences are based on functional layers (Conv, Pool, FC)
    diff_gBest_cp, diff_gBest_fc = _compute_difference(gBest.architecture, particle.architecture)
    diff_pBest_cp, diff_pBest_fc = _compute_difference(particle.pBest_architecture, particle.architecture)

    target_arch_cp = []
    target_arch_fc = []

    # Get current functional layers of the particle
    current_cp = [l for l in particle.architecture if l['type'] in ('conv', 'pool')]
    current_fc = [l for l in particle.architecture if l['type'] == 'fc']

    # Determine target Conv/Pool layers
    max_len_cp = max(len(diff_gBest_cp), len(diff_pBest_cp), len(current_cp))
    for i in range(max_len_cp):
        r = random.random()
        use_gBest_diff = r <= Cg
        source_diff = diff_gBest_cp if use_gBest_diff else diff_pBest_cp

        diff_element = source_diff[i] if i < len(source_diff) else None  # Handle index out of bounds
        current_layer = current_cp[i] if i < len(current_cp) else None

        if diff_element == 0 and current_layer is not None:
            # Keep current layer (type matched)
            target_arch_cp.append(copy.deepcopy(current_layer))
        elif diff_element == -1:
            # Remove layer - do nothing (don't append to target)
            pass
        elif isinstance(diff_element, dict):  # +L case
            # Add/modify to layer L from the difference source (gBest/pBest)
            target_arch_cp.append(copy.deepcopy(diff_element))
        elif diff_element is None and current_layer is not None:
            # Difference vector was shorter than current arch, implies keep if no instruction
            target_arch_cp.append(copy.deepcopy(current_layer))
        # Else (diff_element is None and current_layer is None): Do nothing

    # Determine target FC layers
    max_len_fc = max(len(diff_gBest_fc), len(diff_pBest_fc), len(current_fc))
    for i in range(max_len_fc):
        r = random.random()
        use_gBest_diff = r <= Cg
        source_diff = diff_gBest_fc if use_gBest_diff else diff_pBest_fc

        diff_element = source_diff[i] if i < len(source_diff) else None
        current_layer = current_fc[i] if i < len(current_fc) else None

        if diff_element == 0 and current_layer is not None:
            # Keep current layer (type matched)
            # Special case from Fig 4: If diffs are 0, copy hyperparameters from gBest/pBest
            # Check if *both* diffs were 0 for this position
            gBest_diff_elem = diff_gBest_fc[i] if i < len(diff_gBest_fc) else None
            pBest_diff_elem = diff_pBest_fc[i] if i < len(diff_pBest_fc) else None
            if gBest_diff_elem == 0 and pBest_diff_elem == 0:
                # Special case: Copy hyperparams from chosen source (gBest/pBest)
                source_arch = gBest.architecture if use_gBest_diff else particle.pBest_architecture
                source_fc_layers = [l for l in source_arch if l['type'] == 'fc']
                if i < len(source_fc_layers):
                    target_arch_fc.append(copy.deepcopy(source_fc_layers[i]))
                else:  # Fallback if source is somehow shorter
                    target_arch_fc.append(copy.deepcopy(current_layer))
            else:
                # Normal case: keep current layer as type matched
                target_arch_fc.append(copy.deepcopy(current_layer))

        elif diff_element == -1:
            # Remove layer
            pass
        elif isinstance(diff_element, dict):  # +L case
            # Add/modify to layer L
            target_arch_fc.append(copy.deepcopy(diff_element))
        elif diff_element is None and current_layer is not None:
            # Keep current layer if no instruction
            target_arch_fc.append(copy.deepcopy(current_layer))
        # Else: Do nothing

    # The particle's velocity is set to the list of target functional layers
    particle.velocity = target_arch_cp + target_arch_fc
    # print(f"  Updated velocity (target arch functional layers: {len(particle.velocity)}).") # Debug print

    return particle.velocity


# --- PSO Helper: Particle Update ---

def UpdateParticle(particle, config):
    """
    Updates the particle's architecture based on its computed velocity
    (which represents the target functional architecture).
    Re-inserts BN/Dropout layers based on config and validates the architecture.

    Args:
        particle (Particle): The particle to update.
        config (dict): Configuration dictionary (e.g., for use_bn, n_out).

    Returns:
        Particle: The updated particle.
    """
    target_functional_layers = particle.velocity  # Velocity holds target functional layers

    if not target_functional_layers:  # Velocity computation might have failed or resulted in empty
        print("  Warning: Target architecture (velocity) is empty. Reverting to pBest.")
        particle.architecture = copy.deepcopy(particle.pBest_architecture)
        particle.velocity = []
        return particle

    # Reconstruct full architecture including BN/Dropout based on target functional layers
    new_architecture = []
    use_bn = config.get('use_bn', False)
    use_dropout = config.get('use_dropout', False)
    dropout_rate = config.get('dropout_rate', 0.5)
    n_out = config.get('n_out')  # Needed for final layer check

    current_has_fc = False
    for idx, layer in enumerate(target_functional_layers):
        is_last_functional_layer = (idx == len(target_functional_layers) - 1)
        new_architecture.append(layer)  # Add functional layer
        if layer['type'] == 'fc': current_has_fc = True
        # Add BN
        if use_bn and layer['type'] in ('conv', 'fc') and not is_last_functional_layer:
            new_architecture.append({'type': 'bn'})
        # Add Dropout
        if use_dropout and dropout_rate > 0 and layer['type'] == 'fc' and not is_last_functional_layer:
            new_architecture.append({'type': 'dropout', 'rate': dropout_rate})

    # --- Validation Steps ---
    valid_arch = True
    if not new_architecture:
        print("  ERROR: UpdateParticle resulted in empty architecture.")
        valid_arch = False
    elif new_architecture[0]['type'] != 'conv':
        print("  ERROR: UpdateParticle: First layer must be Conv.")
        valid_arch = False
    elif new_architecture[-1]['type'] != 'fc' or new_architecture[-1].get('neurons') != n_out:
        print(f"  ERROR: UpdateParticle: Last layer must be FC with {n_out} neurons.")
        valid_arch = False
    else:
        # Check FC layer sequencing
        fc_started = False
        functional_layer_idx = -1
        for layer in new_architecture:
            # Only check functional layer sequence
            if layer['type'] in ('conv', 'pool', 'fc'):
                functional_layer_idx += 1
                is_functional_fc = layer['type'] == 'fc'

                if is_functional_fc:
                    fc_started = True
                elif fc_started and not is_functional_fc:  # Found Conv/Pool after FC started
                    print("  ERROR: UpdateParticle: Found Conv/Pool layer after FC layer.")
                    valid_arch = False
                    break
        # Optional: Check pooling constraints (complex, requires input shape)
        # max_pool_layers = calculate_max_pool_layers(...)
        # current_pool_layers = sum(1 for l in new_architecture if l['type'] == 'pool')
        # if current_pool_layers > max_pool_layers: valid_arch = False; print("ERROR: Too many pooling layers.")

    # If architecture is invalid after update, revert to pBest
    if not valid_arch:
        print("  Warning: Invalid architecture after update. Reverting particle to pBest.")
        particle.architecture = copy.deepcopy(particle.pBest_architecture)
        particle.velocity = []  # Reset velocity as it led to invalid state
    else:
        # Update successful
        particle.architecture = new_architecture
        print(f"  Updated particle architecture (Total layers: {len(particle.architecture)}).")

    return particle
