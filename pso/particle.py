import copy


class Particle:
    """
    Represents a single particle in the Particle Swarm Optimization algorithm.

    Each particle encodes a potential CNN architecture and maintains its state
    throughout the optimization process.
    """

    def __init__(self):
        """Initializes a new Particle."""

        # --- Core Components ---
        # Represents the CNN architecture as a list of layer definitions (dictionaries).
        # Example: [{'type': 'conv', 'out_channels': 32, 'kernel_size': 3}, {'type': 'pool', ...}, ...]
        self.architecture = []

        # Represents the particle's velocity. In this context, after refinement,
        # it holds the *target* architecture proposed for the next step,
        # derived from pBest and gBest influences. It's also a list of layer dicts.
        self.velocity = []

        # --- Fitness and History ---
        # The fitness value (loss) of the current architecture. Lower is better.
        # Initialized to infinity.
        self.loss = float('inf')

        # The best architecture found *by this particle* so far (personal best).
        # Stored as a list of layer definitions.
        self.pBest_architecture = []

        # The loss associated with the personal best architecture.
        self.pBest_loss = float('inf')

        # Optional: Store the depth explicitly if useful elsewhere
        # self.depth = 0 # Can be calculated from len(self.architecture)

    def update_pBest(self):
        """
        Updates the particle's personal best if the current state is better.
        """
        if self.loss <= self.pBest_loss:
            # Use deepcopy to avoid modification issues if self.architecture changes later
            self.pBest_architecture = copy.deepcopy(self.architecture)
            self.pBest_loss = self.loss
            # print(f"  Particle updated pBest: Loss {self.pBest_loss:.4f}") # Optional debug print

    def __repr__(self):
        """Provides a string representation of the particle for easy inspection."""
        arch_repr = f"{len(self.architecture)} layers"
        # Could add more details like first/last layer types if needed
        # arch_repr = f"[{self.architecture[0]['type']}...{self.architecture[-1]['type']}] ({len(self.architecture)} layers)" if self.architecture else "[] (0 layers)"

        return (f"Particle(Loss: {self.loss:.4f}, "
                f"pBest Loss: {self.pBest_loss:.4f}, Arch: {arch_repr})")


# --- Example Usage (for testing this file standalone) ---
if __name__ == '__main__':
    p1 = Particle()
    print("Initial Particle:", p1)

    # Simulate finding an architecture and evaluating it
    p1.architecture = [
        {'type': 'conv', 'out_channels': 16, 'kernel_size': 3},
        {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2},
        {'type': 'fc', 'neurons': 10}
    ]
    p1.loss = 1.2345

    print("After setting arch and loss:", p1)

    # Update pBest
    p1.update_pBest()
    print("After updating pBest:", p1)
    print("pBest architecture:", p1.pBest_architecture)

    # Simulate finding a worse architecture
    p1.architecture = p1.architecture[:-1]  # Remove last layer
    p1.loss = 2.5
    p1.update_pBest()  # This should NOT change pBest
    print("After worse loss, pBest loss remains:", p1.pBest_loss)

    # Simulate finding a better architecture
    p1.architecture = [
        {'type': 'conv', 'out_channels': 8, 'kernel_size': 5},
        {'type': 'fc', 'neurons': 10}
    ]
    p1.loss = 0.987
    p1.update_pBest()  # This SHOULD update pBest
    print("After better loss:", p1)
    print("New pBest architecture:", p1.pBest_architecture)
