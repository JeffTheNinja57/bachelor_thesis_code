import random

from .particle import Particle  # Import the Particle class


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


def InitializeSwarm(N, l_max, maps_max, k_max, n_max, n_out,
                    use_bn=False, use_dropout=False, dropout_rate=0.5, max_fc_layers=5):
    """
    Initializes the swarm with N particles having random CNN architectures.
    Optionally adds Batch Normalization and Dropout layers if enabled.
    """
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
        fc_layer_count = 0
        # Generate functional layers first
        for j in range(depth):
            layer_index = j + 1
            if layer_index == 1:
                layer = _add_random_conv(k_max, maps_max)
            elif layer_index == depth:
                layer = {'type': 'fc', 'neurons': n_out}
                has_fc_layer_appeared = True
                fc_layer_count += 1
            elif has_fc_layer_appeared and fc_layer_count < max_fc_layers:
                layer = _add_random_fc(n_max)
                fc_layer_count += 1
            elif has_fc_layer_appeared:
                # If we've reached the FC layer limit, add conv or pool instead
                layer_type_choice = random.randint(1, 2)
                if layer_type_choice == 1:
                    layer = _add_random_conv(k_max, maps_max)
                else:
                    layer = _add_random_pool()
            else:
                # If we haven't reached an FC layer yet, consider the FC layer limit
                if fc_layer_count >= max_fc_layers - 1:  # Reserve one FC for output layer
                    layer_type_choice = random.randint(1, 2)
                    if layer_type_choice == 1:
                        layer = _add_random_conv(k_max, maps_max)
                    else:
                        layer = _add_random_pool()
                else:
                    layer_type_choice = random.randint(1, 3)
                    if layer_type_choice == 1:
                        layer = _add_random_conv(k_max, maps_max)
                    elif layer_type_choice == 2:
                        layer = _add_random_pool()
                    else:
                        layer = _add_random_fc(n_max)
                        has_fc_layer_appeared = True
                        fc_layer_count += 1
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
