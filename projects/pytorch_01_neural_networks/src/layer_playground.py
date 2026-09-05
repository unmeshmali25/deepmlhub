"""
Layer Playground: Experimenting with PyTorch Layers
===================================================

Explore different types of layers and understand their behavior.

Run: python src/layer_playground.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_layers():
    """
    Exercise 1: Linear (Fully Connected) Layers

    Questions:
    - How are weights initialized by default?
    - What's the parameter count formula?
    - When should you use bias=False?
    """
    print("=" * 60)
    print("Exercise 1: Linear Layers")
    print("=" * 60)

    # Create linear layer
    layer = nn.Linear(in_features=10, out_features=5, bias=True)

    print(f"Weight shape: {layer.weight.shape}")  # (out_features, in_features)
    print(f"Bias shape: {layer.bias.shape}")      # (out_features,)

    # Parameter count
    params = layer.weight.numel() + layer.bias.numel()
    formula_params = 10 * 5 + 5
    print(f"\nParameter count: {params}")
    print(f"Formula: (in * out) + out = {formula_params}")

    # Forward pass
    x = torch.randn(32, 10)  # (batch_size, in_features)
    output = layer(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # (32, 5)

    # Weight initialization
    print(f"\nWeight mean: {layer.weight.mean():.4f}")
    print(f"Weight std: {layer.weight.std():.4f}")
    print("Default init: Uniform distribution with smart bounds")

    print("=" * 60)
    print()


def convolutional_layers():
    """
    Exercise 2: Convolutional Layers

    Questions:
    - How does padding affect output size?
    - What's the receptive field?
    - Stride vs dilation: what's the difference?
    """
    print("=" * 60)
    print("Exercise 2: Convolutional Layers")
    print("=" * 60)

    # 2D Convolution (for images)
    conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                     stride=1, padding=1, bias=True)

    print(f"Weight shape: {conv.weight.shape}")  # (out_ch, in_ch, kH, kW)
    print(f"Bias shape: {conv.bias.shape}")      # (out_channels,)

    # Parameter count
    params = conv.weight.numel() + conv.bias.numel()
    formula = (3 * 64 * 3 * 3) + 64
    print(f"Parameters: {params} (formula: {formula})")

    # Forward pass
    x = torch.randn(8, 3, 224, 224)  # (batch, channels, height, width)
    output = conv(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Output size calculation
    # out_size = (in_size + 2*padding - kernel_size) / stride + 1
    print("\nOutput size formula: (H + 2*P - K) / S + 1")
    print(f"Height: (224 + 2*1 - 3) / 1 + 1 = {(224 + 2*1 - 3) // 1 + 1}")

    # Different configurations
    print("\n" + "-" * 60)
    print("Different Conv Configurations:")
    print("-" * 60)

    configs = [
        {"name": "Same padding (stride=1)", "kernel_size": 3, "stride": 1, "padding": 1},
        {"name": "Downsampling (stride=2)", "kernel_size": 3, "stride": 2, "padding": 1},
        {"name": "Large kernel", "kernel_size": 7, "stride": 2, "padding": 3},
        {"name": "Dilated conv", "kernel_size": 3, "stride": 1, "padding": 2, "dilation": 2},
    ]

    for cfg in configs:
        name = cfg.pop("name")
        layer = nn.Conv2d(in_channels=3, out_channels=64, **cfg)
        out = layer(x)
        print(f"{name}: {x.shape} -> {out.shape}")

    print("=" * 60)
    print()


def normalization_layers():
    """
    Exercise 3: Normalization Layers

    Questions:
    - BatchNorm vs LayerNorm vs InstanceNorm: when to use each?
    - Why does BatchNorm have different behavior in train vs eval mode?
    - What are affine parameters (gamma, beta)?
    """
    print("=" * 60)
    print("Exercise 3: Normalization Layers")
    print("=" * 60)

    batch_size = 16
    channels = 64
    height = width = 32

    x = torch.randn(batch_size, channels, height, width)
    print(f"Input shape: {x.shape}\n")

    # BatchNorm2d
    bn = nn.BatchNorm2d(num_features=channels, affine=True)
    out_bn = bn(x)

    print("1. BatchNorm2d")
    print(f"   Normalizes across batch dimension for each channel")
    print(f"   Output shape: {out_bn.shape}")
    print(f"   Learnable params: gamma (weight), beta (bias)")
    print(f"   Weight shape: {bn.weight.shape}")
    print(f"   Running mean shape: {bn.running_mean.shape}")

    # LayerNorm
    ln = nn.LayerNorm(normalized_shape=[channels, height, width])
    out_ln = ln(x)

    print("\n2. LayerNorm")
    print(f"   Normalizes across channel/spatial dimensions for each sample")
    print(f"   Output shape: {out_ln.shape}")
    print(f"   Useful for: RNNs, Transformers (batch-size independent)")

    # InstanceNorm2d
    in_norm = nn.InstanceNorm2d(num_features=channels, affine=False)
    out_in = in_norm(x)

    print("\n3. InstanceNorm2d")
    print(f"   Normalizes each channel independently per sample")
    print(f"   Output shape: {out_in.shape}")
    print(f"   Useful for: Style transfer, GANs")

    # GroupNorm
    gn = nn.GroupNorm(num_groups=8, num_channels=channels)
    out_gn = gn(x)

    print("\n4. GroupNorm")
    print(f"   Divides channels into groups and normalizes within groups")
    print(f"   Output shape: {out_gn.shape}")
    print(f"   Useful for: Small batch sizes (batch-size independent)")

    # Train vs Eval mode for BatchNorm
    print("\n" + "-" * 60)
    print("BatchNorm: Train vs Eval Mode")
    print("-" * 60)

    bn = nn.BatchNorm2d(64)
    x = torch.randn(16, 64, 32, 32)

    bn.train()
    out_train = bn(x)
    running_mean_train = bn.running_mean.clone()

    bn.eval()
    out_eval = bn(x)
    running_mean_eval = bn.running_mean.clone()

    print(f"Train mode: Uses batch statistics, updates running stats")
    print(f"  Running mean changed: {not torch.equal(running_mean_train, running_mean_eval)}")
    print(f"\nEval mode: Uses running statistics (no update)")
    print(f"  Important for inference!")

    print("=" * 60)
    print()


def pooling_layers():
    """
    Exercise 4: Pooling Layers

    Questions:
    - MaxPool vs AvgPool: when to use each?
    - What happens to gradient flow through pooling?
    - AdaptiveAvgPool: how does it work?
    """
    print("=" * 60)
    print("Exercise 4: Pooling Layers")
    print("=" * 60)

    x = torch.randn(8, 64, 32, 32)
    print(f"Input shape: {x.shape}\n")

    # MaxPool2d
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    out_max = maxpool(x)

    print("1. MaxPool2d (kernel=2, stride=2)")
    print(f"   Output shape: {out_max.shape}")
    print(f"   Downsamples by factor of 2")
    print(f"   Keeps maximum value in each window")

    # AvgPool2d
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    out_avg = avgpool(x)

    print("\n2. AvgPool2d (kernel=2, stride=2)")
    print(f"   Output shape: {out_avg.shape}")
    print(f"   Computes average in each window")
    print(f"   Smoother gradients than MaxPool")

    # AdaptiveAvgPool2d
    adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Global Average Pooling
    out_adaptive = adaptive_pool(x)

    print("\n3. AdaptiveAvgPool2d (output_size=(1,1))")
    print(f"   Output shape: {out_adaptive.shape}")
    print(f"   Automatically computes kernel size to achieve target output")
    print(f"   Useful for: Making CNNs work with variable input sizes")

    # Global Average Pooling (common in modern architectures)
    gap = nn.AdaptiveAvgPool2d((1, 1))
    out_gap = gap(x)
    out_gap_flat = out_gap.view(out_gap.size(0), -1)

    print("\n4. Global Average Pooling (GAP)")
    print(f"   Before GAP: {x.shape}")
    print(f"   After GAP: {out_gap.shape}")
    print(f"   After flatten: {out_gap_flat.shape}")
    print(f"   Replaces fully connected layers in modern CNNs (e.g., ResNet)")

    print("=" * 60)
    print()


def dropout_and_regularization():
    """
    Exercise 5: Dropout and Regularization

    Questions:
    - How does dropout prevent overfitting?
    - Why scale activations by 1/p during training?
    - Dropout vs Dropout2d: what's the difference?
    """
    print("=" * 60)
    print("Exercise 5: Dropout and Regularization")
    print("=" * 60)

    # Standard Dropout
    dropout = nn.Dropout(p=0.5)
    x = torch.ones(4, 10)

    print("1. Dropout (p=0.5)")
    print(f"   Input:\n{x[0]}")

    dropout.train()
    out_train = dropout(x)
    print(f"\n   Train mode output:\n{out_train[0]}")
    print(f"   ~50% of values zeroed, rest scaled by 2")

    dropout.eval()
    out_eval = dropout(x)
    print(f"\n   Eval mode output:\n{out_eval[0]}")
    print(f"   No dropout applied (identity function)")

    # Dropout2d (for CNNs)
    print("\n2. Dropout2d (for CNNs)")
    dropout2d = nn.Dropout2d(p=0.5)
    x_conv = torch.randn(8, 64, 32, 32)

    dropout2d.train()
    out_conv = dropout2d(x_conv)

    print(f"   Input shape: {x_conv.shape}")
    print(f"   Drops entire feature maps (channels), not individual pixels")
    print(f"   Useful after convolutional layers")

    # Compare with regular Dropout
    print("\n   Key Difference:")
    print(f"   - Dropout: Drops individual elements")
    print(f"   - Dropout2d: Drops entire channels (feature maps)")

    print("=" * 60)
    print()


def activation_functions():
    """
    Exercise 6: Activation Functions

    Questions:
    - ReLU vs LeakyReLU vs ELU: what's the difference?
    - Why is ReLU preferred over Sigmoid/Tanh in deep networks?
    - What is the "dying ReLU" problem?
    """
    print("=" * 60)
    print("Exercise 6: Activation Functions")
    print("=" * 60)

    x = torch.linspace(-5, 5, 100)

    activations = {
        "ReLU": nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(negative_slope=0.01),
        "ELU": nn.ELU(alpha=1.0),
        "GELU": nn.GELU(),
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh(),
    }

    print(f"{'Activation':<15} {'Range':<20} {'Use Case'}")
    print("-" * 60)

    for name, act in activations.items():
        out = act(x)
        min_val, max_val = out.min().item(), out.max().item()
        range_str = f"[{min_val:.2f}, {max_val:.2f}]"

        use_cases = {
            "ReLU": "CNNs, MLPs (default choice)",
            "LeakyReLU": "Fixes dying ReLU problem",
            "ELU": "Smoother gradients than ReLU",
            "GELU": "Transformers (BERT, GPT)",
            "Sigmoid": "Binary classification output",
            "Tanh": "RNNs, LSTMs",
        }

        print(f"{name:<15} {range_str:<20} {use_cases[name]}")

    print("\nKey Points:")
    print("- ReLU: Fast, but can 'die' (always output 0)")
    print("- LeakyReLU: Allows small gradient when x < 0")
    print("- GELU: Smooth, probabilistic (used in modern Transformers)")
    print("- Sigmoid/Tanh: Vanishing gradient problem in deep networks")

    print("=" * 60)
    print()


def main():
    """Run all layer demonstrations."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}\n")

    linear_layers()
    convolutional_layers()
    normalization_layers()
    pooling_layers()
    dropout_and_regularization()
    activation_functions()

    print("\n" + "=" * 60)
    print("PRACTICE QUESTIONS:")
    print("=" * 60)
    print("""
    1. Calculate output size for: Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
       with input (8, 3, 128, 128).

    2. Build a module with Conv -> BatchNorm -> ReLU -> MaxPool. Test on image batch.
       Observe how BatchNorm statistics change between train and eval mode.

    3. Implement depthwise separable convolution (used in MobileNet):
       - Depthwise: Conv2d(C, C, groups=C)
       - Pointwise: Conv2d(C, C, kernel_size=1)
       Compare parameter count with regular Conv2d.

    4. Create a residual block with different normalization types (BatchNorm,
       LayerNorm, GroupNorm). Compare training behavior.

    5. Test dropout with different values of p (0.1, 0.5, 0.9). How does it
       affect the output distribution during training?

    6. Implement Squeeze-and-Excitation (SE) block:
       - Global Average Pooling
       - Two Linear layers (compression + expansion)
       - Sigmoid activation
       - Multiply with input (channel attention)

    7. Build a "Spatial Transformer" layer that applies learned affine
       transformation to input feature maps.
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()
