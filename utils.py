import numpy as np

try:
    # Python 2 module
    from itertools import izip_longest as zip_longest
except ImportError:
    # Python 3 module
    from itertools import zip_longest


def add_conv(layers, max_out_ch, conv_kernel):
    """
    Add a convolutional layer to the network architecture.

    Args:
        layers (list): Current list of layers in the network
        max_out_ch (int): Maximum number of output channels
        conv_kernel (int): Maximum kernel size for convolution

    Returns:
        list: Updated list of layers with the new convolutional layer added
    """
    out_channel = np.random.randint(3, max_out_ch)
    conv_kernel = np.random.randint(3, conv_kernel)

    layers.append({"type": "conv", "ou_c": out_channel, "kernel": conv_kernel})

    return layers


def add_res(layers, max_out_ch, conv_kernel):
    """
    Add a residual layer to the network architecture.

    Args:
        layers (list): Current list of layers in the network
        max_out_ch (int): Maximum number of output channels
        conv_kernel (int): Maximum kernel size for convolution

    Returns:
        list: Updated list of layers with the new residual layer added
    """
    out_channel = np.random.randint(3, max_out_ch)
    conv_kernel = np.random.randint(3, conv_kernel)

    layers.append({"type": "res", "ou_c": out_channel, "kernel": conv_kernel})

    return layers


def add_fc(layers, max_fc_neurons):
    """
    Add a fully connected layer to the network architecture.

    Args:
        layers (list): Current list of layers in the network
        max_fc_neurons (int): Maximum number of neurons in the fully connected layer

    Returns:
        list: Updated list of layers with the new fully connected layer added
    """
    layers.append({"type": "fc", "ou_c": np.random.randint(1, max_fc_neurons), "kernel": -1})

    return layers


def add_pool(layers, fc_prob, num_pool_layers, max_pool_layers, max_out_ch, max_conv_kernel, max_fc_neurons,
             output_dim):
    """
    Add a pooling layer to the network architecture if the maximum number of pooling layers has not been reached.
    Randomly selects between max pooling and average pooling.

    Args:
        layers (list): Current list of layers in the network
        fc_prob (float): Probability of adding a fully connected layer
        num_pool_layers (int): Current number of pooling layers in the network
        max_pool_layers (int): Maximum number of pooling layers allowed
        max_out_ch (int): Maximum number of output channels
        max_conv_kernel (int): Maximum kernel size for convolution
        max_fc_neurons (int): Maximum number of neurons in fully connected layers
        output_dim (int): Output dimension of the network

    Returns:
        tuple: (updated list of layers, updated count of pooling layers)
    """
    pool_layers = num_pool_layers

    if pool_layers < max_pool_layers:
        random_pool = np.random.rand()
        pool_layers += 1
        if random_pool < 0.5:
            # Add Max Pooling
            layers.append({"type": "max_pool", "ou_c": -1, "kernel": 2})
        else:
            layers.append({"type": "avg_pool", "ou_c": -1, "kernel": 2})

    return layers, pool_layers


def differenceConvPool(p1, p2):
    """
    Compute the difference between convolutional and pooling layers of two network architectures.
    Used in the PSO algorithm to determine how architectures differ.

    Args:
        p1 (list): First particle's convolutional and pooling layers
        p2 (list): Second particle's convolutional and pooling layers

    Returns:
        list: Difference representation between the two architectures' conv/pool layers
              with "keep" for identical layers, original layer from p1 for different layers,
              and "remove" for layers that exist in p2 but not in p1
    """
    diff = []

    for comb in zip_longest(p1, p2):
        if comb[0] != None and comb[1] != None:
            if comb[0]["type"] == comb[1]["type"]:
                diff.append({"type": "keep"})
            else:
                diff.append(comb[0])

        elif comb[0] != None and comb[1] == None:
            diff.append(comb[0])

        elif comb[0] == None and comb[1] != None:
            diff.append({"type": "remove"})

    return diff


def differenceFC(p1, p2):
    """
    Compute the difference between fully connected layers of two network architectures.
    Used in the PSO algorithm to determine how architectures differ.

    Args:
        p1 (list): First particle's fully connected layers
        p2 (list): Second particle's fully connected layers

    Returns:
        list: Difference representation between the two architectures' FC layers
              with "keep_fc" for identical layers, original layer from p1 for different layers,
              and "remove_fc" for layers that exist in p2 but not in p1

    Note:
        This function processes the layers in reverse order (from end to beginning)
        and then reverses the result back to maintain the correct layer order.
    """
    diff = []

    # Compute the difference from the end to the begin
    for comb in zip_longest(p1[::-1], p2[::-1]):
        if comb[0] != None and comb[1] != None:
            diff.append({"type": "keep_fc"})
        elif comb[0] != None and comb[1] == None:
            diff.append(comb[0])
        elif comb[0] == None and comb[1] != None:
            diff.append({"type": "remove_fc"})

    diff = diff[::-1]

    return diff


def computeDifference(p1, p2):
    """
    Compute the overall difference between two network architectures (particles).
    This function handles both convolutional/pooling layers and fully connected layers.

    Args:
        p1 (list): First particle's complete architecture
        p2 (list): Second particle's complete architecture

    Returns:
        tuple: (difference representation, boolean indicating if all layers are kept)
               The difference representation is a list showing how p1 differs from p2,
               and the boolean indicates if all layers in both architectures are identical.
    """
    diff = []
    # First, find the index where the fully connected layers start in each particle
    p1fc_idx = next((index for (index, d) in enumerate(p1) if d["type"] == "fc"))
    p2fc_idx = next((index for (index, d) in enumerate(p2) if d["type"] == "fc"))

    # Compute the difference only between the convolution and pooling layers
    diff.extend(differenceConvPool(p1[0:p1fc_idx], p2[0:p2fc_idx]))

    # Compute the difference between the fully connected layers 
    diff.extend(differenceFC(p1[p1fc_idx:], p2[p2fc_idx:]))

    keep_all_layers = True
    for i in range(len(diff)):
        if diff[i]["type"] != "keep" or diff[i]["type"] != "keep_fc":
            keep_all_layers = False
            break

    return diff, keep_all_layers


def velocityConvPool(diff_pBest, diff_gBest, Cg):
    """
    Calculate the velocity for convolutional and pooling layers in PSO algorithm.
    This function determines which architectural changes to apply based on personal
    and global best differences, with probability Cg.

    Args:
        diff_pBest (list): Difference between personal best and current architecture
        diff_gBest (list): Difference between global best and current architecture
        Cg (float): Probability of choosing global best difference over personal best

    Returns:
        list: Velocity representation for convolutional and pooling layers
    """
    vel = []

    for comb in zip_longest(diff_pBest, diff_gBest):
        if np.random.uniform() <= Cg:
            if comb[1] != None:
                vel.append(comb[1])
            else:
                vel.append({"type": "remove"})
        else:
            if comb[0] != None:
                vel.append(comb[0])
            else:
                vel.append({"type": "remove"})

    return vel


def velocityFC(diff_pBest, diff_gBest, Cg):
    """
    Calculate the velocity for fully connected layers in PSO algorithm.
    This function determines which architectural changes to apply based on personal
    and global best differences, with probability Cg.

    Args:
        diff_pBest (list): Difference between personal best and current architecture for FC layers
        diff_gBest (list): Difference between global best and current architecture for FC layers
        Cg (float): Probability of choosing global best difference over personal best

    Returns:
        list: Velocity representation for fully connected layers

    Note:
        This function processes the layers in reverse order (from end to beginning)
        and then reverses the result back to maintain the correct layer order.
    """
    vel = []

    for comb in zip_longest(diff_pBest[::-1], diff_gBest[::-1]):
        if np.random.uniform() <= Cg:
            if comb[1] != None:
                vel.append(comb[1])
            else:
                vel.append({"type": "remove_fc"})
        else:
            if comb[0] != None:
                vel.append(comb[0])
            else:
                vel.append({"type": "remove_fc"})

    vel = vel[::-1]

    return vel


def computeVelocity(gBest, pBest, p, Cg):
    """
    Compute the overall velocity for a particle in the PSO algorithm.
    This function determines how a particle's architecture should change based on
    its personal best and the global best architectures.

    Args:
        gBest (list): Global best architecture
        pBest (list): Personal best architecture for this particle
        p (list): Current architecture of the particle
        Cg (float): Probability of choosing global best over personal best

    Returns:
        list: Velocity representation for the entire architecture
    """
    diff_pBest, keep_all_pBest = computeDifference(pBest, p)
    diff_gBest, keep_all_gBest = computeDifference(gBest, p)

    velocity = []

    # First, verify if the general architecture is the same in both difference
    if keep_all_pBest == True and keep_all_gBest == True:
        for i in range(len(gBest)):
            if np.random.uniform() <= Cg:
                velocity.append(gBest[i])
            else:
                velocity.append(pBest[i])
    else:
        # Find the index where the fully connected layers start in each particle
        dp_fc_idx = next((index for (index, d) in enumerate(diff_pBest) if
                          d["type"] == "fc" or d["type"] == "keep_fc" or d["type"] == "remove_fc"))
        dg_fc_idx = next((index for (index, d) in enumerate(diff_gBest) if
                          d["type"] == "fc" or d["type"] == "keep_fc" or d["type"] == "remove_fc"))

        # Compute the velocity only between the convolution and pooling layers
        velocity.extend(velocityConvPool(diff_pBest[0:dp_fc_idx], diff_gBest[0:dg_fc_idx], Cg))

        # Compute the velocity between the fully connected layers
        velocity.extend(velocityFC(diff_pBest[dp_fc_idx:], diff_gBest[dg_fc_idx:], Cg))

    return velocity


def updateConvPool(p, vel):
    """
    Update the convolutional and pooling layers of a particle based on the calculated velocity.

    Args:
        p (list): Current convolutional and pooling layers of the particle
        vel (list): Velocity representation for convolutional and pooling layers

    Returns:
        list: Updated convolutional and pooling layers
    """
    new_p = []

    for comb in zip_longest(p, vel):
        if comb[1]["type"] != "remove":
            if comb[1]["type"] == "keep":
                new_p.append(comb[0])
            else:
                new_p.append(comb[1])

    return new_p


def updateFC(p, vel):
    """
    Update the fully connected layers of a particle based on the calculated velocity.

    Args:
        p (list): Current fully connected layers of the particle
        vel (list): Velocity representation for fully connected layers

    Returns:
        list: Updated fully connected layers

    Note:
        This function processes the layers in reverse order (from end to beginning)
        and then reverses the result back to maintain the correct layer order.
    """
    new_p = []

    for comb in zip_longest(p[::-1], vel[::-1]):
        if comb[1]["type"] != "remove_fc":
            if comb[1]["type"] == "keep_fc":
                new_p.append(comb[0])
            else:
                new_p.append(comb[1])

    new_p = new_p[::-1]

    return new_p


def updateParticle(p, velocity):
    """
    Update a particle's entire architecture based on the calculated velocity.
    This function handles both convolutional/pooling layers and fully connected layers.

    Args:
        p (list): Current architecture of the particle
        velocity (list): Velocity representation for the entire architecture

    Returns:
        list: Updated architecture for the particle
    """
    new_p = []

    dp_fc_idx = next((index for (index, d) in enumerate(p) if d["type"] == "fc"))
    dg_fc_idx = next((index for (index, d) in enumerate(velocity) if
                      d["type"] == "fc" or d["type"] == "keep_fc" or d["type"] == "remove_fc"))

    # Update only convolution and pooling layers
    new_p.extend(updateConvPool(p[0:dp_fc_idx], velocity[0:dg_fc_idx]))

    # Update only fully connected layers
    new_p.extend(updateFC(p[dp_fc_idx:], velocity[dg_fc_idx:]))

    return new_p
