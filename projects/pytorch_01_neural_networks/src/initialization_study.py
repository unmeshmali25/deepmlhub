"""
Weight Initialization Strategies
==================================

Learn how weight initialization affects training convergence.
Explore different initialization schemes and their impact.

Run: python src/initialization_study.py
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def demonstrate_initialization_methods():
    """
    Exercise 1: Different Initialization Methods

    Questions:
    - What's the purpose of weight initialization?
    - Why not initialize all weights to zero?
    - What's the difference between Xavier and Kaiming initialization?
    """
    print("=" * 60)
    print("Exercise 1: Initialization Methods")
    print("=" * 60)

    # Create a layer (don't use built-in init yet)
    layer_size = (1000, 1000)

    methods = {
        "Zeros": lambda: torch.zeros(layer_size),
        "Ones": lambda: torch.ones(layer_size),
        "Constant (0.01)": lambda: torch.full(layer_size, 0.01),
        "Uniform [-0.1, 0.1]": lambda: torch.empty(layer_size).uniform_(-0.1, 0.1),
        "Normal (0, 0.01)": lambda: torch.empty(layer_size).normal_(0, 0.01),
        "Xavier Uniform": lambda: init.xavier_uniform_(torch.empty(layer_size)),
        "Xavier Normal": lambda: init.xavier_normal_(torch.empty(layer_size)),
        "Kaiming Uniform": lambda: init.kaiming_uniform_(torch.empty(layer_size), nonlinearity='relu'),
        "Kaiming Normal": lambda: init.kaiming_normal_(torch.empty(layer_size), nonlinearity='relu'),
    }

    print(f"{'Method':<20} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-" * 80)

    for name, init_fn in methods.items():
        w = init_fn()
        print(f"{name:<20} {w.mean():.6f}{'':<7} {w.std():.6f}{'':<7} "
              f"{w.min():.6f}{'':<7} {w.max():.6f}")

    print("\nKey Insights:")
    print("- All zeros: Symmetry problem (all neurons learn the same thing)")
    print("- Xavier: For tanh/sigmoid activations (variance preservation)")
    print("- Kaiming (He): For ReLU activations (accounts for ReLU zeroing)")

    print("=" * 60)
    print()


def variance_preservation():
    """
    Exercise 2: Variance Preservation Through Layers

    Questions:
    - What happens to activation variance with bad initialization?
    - Why does Xavier initialization preserve variance?
    - How many layers can you stack before variance explodes/vanishes?
    """
    print("=" * 60)
    print("Exercise 2: Variance Preservation")
    print("=" * 60)

    input_size = 1000
    hidden_size = 1000
    num_layers = 10
    batch_size = 128

    # Test input
    x = torch.randn(batch_size, input_size)

    # Different initialization strategies
    init_strategies = {
        "No init (default)": None,
        "Xavier": "xavier",
        "Kaiming": "kaiming",
        "Small random": "small",
    }

    results = {}

    for strategy_name, strategy in init_strategies.items():
        # Build network
        layers = []
        for _ in range(num_layers):
            layer = nn.Linear(hidden_size if layers else input_size, hidden_size)

            # Apply initialization
            if strategy == "xavier":
                init.xavier_normal_(layer.weight)
                init.zeros_(layer.bias)
            elif strategy == "kaiming":
                init.kaiming_normal_(layer.weight, nonlinearity='relu')
                init.zeros_(layer.bias)
            elif strategy == "small":
                init.normal_(layer.weight, mean=0, std=0.01)
                init.zeros_(layer.bias)
            # else: use default PyTorch initialization

            layers.append(layer)

        # Forward pass and track variance
        variances = []
        activation = x
        variances.append(activation.var().item())

        for layer in layers:
            activation = layer(activation)
            activation = torch.relu(activation)  # ReLU activation
            variances.append(activation.var().item())

        results[strategy_name] = variances

    # Print results
    print(f"\n{'Layer':<10}", end="")
    for name in init_strategies.keys():
        print(f"{name:<20}", end="")
    print()
    print("-" * 90)

    for i in range(min(num_layers + 1, 11)):  # Show first 11 layers
        print(f"{i:<10}", end="")
        for name in init_strategies.keys():
            var = results[name][i] if i < len(results[name]) else 0
            print(f"{var:<20.6f}", end="")
        print()

    # Plot
    plt.figure(figsize=(10, 6))
    for name, variances in results.items():
        plt.plot(variances, marker='o', label=name, linewidth=2)

    plt.xlabel('Layer Depth')
    plt.ylabel('Activation Variance')
    plt.title('Variance Preservation Through Layers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('projects/pytorch_01_neural_networks/variance_preservation.png',
                dpi=150, bbox_inches='tight')
    print("\nPlot saved: variance_preservation.png")

    print("\nKey Insight:")
    print("- Bad init → variance explodes or vanishes → gradients explode/vanish")
    print("- Xavier/Kaiming → variance stays roughly constant → stable training")

    print("=" * 60)
    print()


def orthogonal_initialization():
    """
    Exercise 3: Orthogonal Initialization

    Questions:
    - What is orthogonal initialization?
    - When is it useful?
    - How does it preserve gradient flow?
    """
    print("=" * 60)
    print("Exercise 3: Orthogonal Initialization")
    print("=" * 60)

    # Create orthogonal weight matrix
    w = torch.empty(100, 100)
    init.orthogonal_(w)

    # Verify orthogonality: W @ W^T = I
    product = w @ w.t()
    identity = torch.eye(100)

    print("Orthogonal matrix properties:")
    print(f"Shape: {w.shape}")
    print(f"Mean: {w.mean():.6f}")
    print(f"Std: {w.std():.6f}")

    print(f"\nW @ W^T ≈ I (identity)?")
    print(f"Max deviation from identity: {(product - identity).abs().max():.6f}")
    print(f"Frobenius norm of (W@W^T - I): {(product - identity).norm():.6f}")

    # Compare with random initialization
    w_random = torch.randn(100, 100) * 0.01
    product_random = w_random @ w_random.t()

    print(f"\nFor random init:")
    print(f"Max deviation from identity: {(product_random - identity).abs().max():.6f}")

    print("\nUse Cases:")
    print("- RNNs: Prevents gradient vanishing/explosion")
    print("- Very deep networks: Preserves gradient flow")
    print("- Initialization for symmetric matrices")

    print("=" * 60)
    print()


def custom_initialization():
    """
    Exercise 4: Custom Initialization Patterns

    Questions:
    - How to initialize specific layer types differently?
    - How to initialize based on layer position in network?
    - How to initialize with pre-trained weights?
    """
    print("=" * 60)
    print("Exercise 4: Custom Initialization")
    print("=" * 60)

    class CustomInitNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.fc1 = nn.Linear(64 * 32 * 32, 512)
            self.fc2 = nn.Linear(512, 10)

        def _init_weights(self):
            """Custom initialization for different layer types."""
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    # Kaiming init for conv layers (good with ReLU)
                    init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        init.constant_(module.bias, 0)

                elif isinstance(module, nn.BatchNorm2d):
                    # Standard practice for BN
                    init.constant_(module.weight, 1)
                    init.constant_(module.bias, 0)

                elif isinstance(module, nn.Linear):
                    # Xavier init for linear layers
                    init.xavier_uniform_(module.weight)
                    init.constant_(module.bias, 0)

    model = CustomInitNet()
    model._init_weights()

    print("Custom initialization applied to:")
    print("- Conv layers: Kaiming Normal (for ReLU)")
    print("- BatchNorm: weight=1, bias=0")
    print("- Linear: Xavier Uniform")

    print("\nModel architecture:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            print(f"  {name}: {module.__class__.__name__}")

    print("=" * 60)
    print()


def bias_initialization():
    """
    Exercise 5: Bias Initialization

    Questions:
    - Should biases be initialized to zero?
    - When to initialize biases to non-zero values?
    - How does bias affect gradient flow?
    """
    print("=" * 60)
    print("Exercise 5: Bias Initialization")
    print("=" * 60)

    # Different bias initialization strategies
    layer1 = nn.Linear(10, 10)
    init.zeros_(layer1.bias)  # Standard: zero

    layer2 = nn.Linear(10, 10)
    init.constant_(layer2.bias, 0.01)  # Small positive constant

    layer3 = nn.Linear(10, 10)
    init.uniform_(layer3.bias, -0.1, 0.1)  # Uniform random

    print("Common bias initialization strategies:")
    print("\n1. Zero (most common):")
    print(f"   Bias: {layer1.bias[:5]}")
    print("   Safe default, doesn't introduce asymmetry")

    print("\n2. Small positive constant (e.g., 0.01):")
    print(f"   Bias: {layer2.bias[:5]}")
    print("   Can help with ReLU dead neurons initially")

    print("\n3. Random uniform:")
    print(f"   Bias: {layer3.bias[:5]}")
    print("   Less common, breaks symmetry more")

    # Special case: LSTM forget gate bias
    print("\n4. Special case - LSTM forget gate:")
    lstm = nn.LSTM(input_size=10, hidden_size=20)
    # Initialize forget gate bias to 1.0 (helps learning long-term dependencies)
    for names in lstm._all_weights:
        for name in names:
            if 'bias' in name:
                bias = getattr(lstm, name)
                n = bias.size(0)
                # LSTM has 4 gates: input, forget, cell, output
                # Forget gate is the second quarter
                bias.data[n // 4:n // 2].fill_(1.0)

    print("   LSTM forget gate bias initialized to 1.0")
    print("   Helps model remember long sequences initially")

    print("=" * 60)
    print()


def initialization_impact_on_training():
    """
    Exercise 6: Impact on Training Speed

    Questions:
    - How does initialization affect convergence speed?
    - Can you train with all-zero initialization?
    - What happens with very large initial weights?
    """
    print("=" * 60)
    print("Exercise 6: Initialization Impact on Training")
    print("=" * 60)

    # Simple network
    def create_network(init_method):
        model = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        # Apply initialization
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if init_method == "zeros":
                    init.zeros_(module.weight)
                    init.zeros_(module.bias)
                elif init_method == "xavier":
                    init.xavier_normal_(module.weight)
                    init.zeros_(module.bias)
                elif init_method == "kaiming":
                    init.kaiming_normal_(module.weight, nonlinearity='relu')
                    init.zeros_(module.bias)
                elif init_method == "large":
                    init.normal_(module.weight, mean=0, std=10.0)
                    init.zeros_(module.bias)

        return model

    # Simulate forward pass with random data
    x = torch.randn(32, 100)
    y_true = torch.randint(0, 10, (32,))

    methods = ["zeros", "xavier", "kaiming", "large"]
    criterion = nn.CrossEntropyLoss()

    print(f"{'Method':<15} {'Initial Loss':<20} {'Max Activation':<20}")
    print("-" * 60)

    for method in methods:
        model = create_network(method)
        model.eval()

        with torch.no_grad():
            output = model(x)
            loss = criterion(output, y_true)
            max_act = output.abs().max()

        status = ""
        if method == "zeros":
            status = "(symmetry - won't learn!)"
        elif method == "large":
            status = "(may explode)"

        print(f"{method:<15} {loss.item():<20.4f} {max_act.item():<20.4f}  {status}")

    print("\nKey Insights:")
    print("- Zeros: Symmetry problem → all neurons learn identical features")
    print("- Large init: Can cause exploding activations/gradients")
    print("- Xavier/Kaiming: Balanced, stable convergence")

    print("=" * 60)
    print()


def main():
    """Run all initialization demonstrations."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}\n")

    demonstrate_initialization_methods()
    variance_preservation()
    orthogonal_initialization()
    custom_initialization()
    bias_initialization()
    initialization_impact_on_training()

    print("\n" + "=" * 60)
    print("PRACTICE QUESTIONS:")
    print("=" * 60)
    print("""
    1. Implement a function that initializes a network based on activation type:
       - ReLU → Kaiming
       - Tanh/Sigmoid → Xavier
       - GELU → ?

    2. Create a 50-layer network. Compare training with:
       - Default initialization
       - Xavier initialization
       - Orthogonal initialization
       Which converges fastest? Why?

    3. Implement "LSUV initialization" (Layer-Sequential Unit-Variance):
       Initialize each layer to have unit variance after forward pass.

    4. Research "Fixup initialization" for ResNets (no BatchNorm needed).
       Implement and test on a residual network.

    5. Design an initialization scheme for a network with skip connections.
       How should you initialize the residual branch vs main branch?

    6. Investigate sparse initialization: Initialize only a fraction of weights
       to non-zero values. How does sparsity affect training?

    7. For a Transformer model, research standard initialization practices:
       - Attention weights
       - Feedforward layers
       - Layer normalization
       Implement and explain the rationale.
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()
