"""
Hooks and Gradient Inspection
==============================

Learn how to use hooks to inspect activations and gradients.
Essential for debugging and understanding deep networks.

Run: python src/hooks_and_gradients.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_hooks():
    """
    Exercise 1: Forward Hooks

    Questions:
    - What are forward hooks used for?
    - How do you access intermediate activations?
    - When is the hook function called?
    """
    print("=" * 60)
    print("Exercise 1: Forward Hooks")
    print("=" * 60)

    # Simple network
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )

    # Storage for activations
    activations = {}

    def get_activation(name):
        """Returns a hook function that stores activations."""
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks
    model[0].register_forward_hook(get_activation('layer1'))
    model[2].register_forward_hook(get_activation('layer2'))
    model[4].register_forward_hook(get_activation('layer3'))

    # Forward pass
    x = torch.randn(8, 10)
    output = model(x)

    print("Activations captured:")
    for name, act in activations.items():
        print(f"  {name}: shape {act.shape}, mean {act.mean():.4f}, std {act.std():.4f}")

    print("\nUse cases:")
    print("- Feature extraction from intermediate layers")
    print("- Debugging activation distributions")
    print("- Implementing attention visualization")

    print("=" * 60)
    print()


def backward_hooks():
    """
    Exercise 2: Backward Hooks

    Questions:
    - What are backward hooks used for?
    - How do you access gradients during backpropagation?
    - What's the difference between grad_input and grad_output?
    """
    print("=" * 60)
    print("Exercise 2: Backward Hooks")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )

    # Storage for gradients
    gradients = {}

    def get_gradient(name):
        """Returns a hook function that stores gradients."""
        def hook(module, grad_input, grad_output):
            # grad_input: gradient w.r.t. inputs to this layer
            # grad_output: gradient w.r.t. outputs of this layer
            gradients[name] = {
                'grad_input': grad_input[0].detach() if grad_input[0] is not None else None,
                'grad_output': grad_output[0].detach()
            }
        return hook

    # Register backward hooks
    model[0].register_full_backward_hook(get_gradient('layer1'))
    model[2].register_full_backward_hook(get_gradient('layer2'))
    model[4].register_full_backward_hook(get_gradient('layer3'))

    # Forward + backward pass
    x = torch.randn(8, 10, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()

    print("Gradients captured:")
    for name, grads in gradients.items():
        grad_out = grads['grad_output']
        print(f"\n  {name}:")
        print(f"    grad_output: shape {grad_out.shape}, norm {grad_out.norm():.4f}")
        if grads['grad_input'] is not None:
            grad_in = grads['grad_input']
            print(f"    grad_input: shape {grad_in.shape}, norm {grad_in.norm():.4f}")

    print("\nUse cases:")
    print("- Detecting vanishing/exploding gradients")
    print("- Implementing gradient clipping per layer")
    print("- Analyzing gradient flow in deep networks")

    print("=" * 60)
    print()


def tensor_hooks():
    """
    Exercise 3: Tensor Hooks (register_hook)

    Questions:
    - How are tensor hooks different from module hooks?
    - When would you use tensor hooks vs module hooks?
    - How do you modify gradients during backprop?
    """
    print("=" * 60)
    print("Exercise 3: Tensor Hooks")
    print("=" * 60)

    # Register hook on a specific tensor
    x = torch.randn(3, 3, requires_grad=True)
    w = torch.randn(3, 3, requires_grad=True)

    gradient_info = []

    def hook_fn(grad):
        """Hook function called during backward pass."""
        gradient_info.append({
            'shape': grad.shape,
            'norm': grad.norm().item(),
            'mean': grad.mean().item()
        })
        print(f"  Gradient hook called!")
        print(f"    Shape: {grad.shape}")
        print(f"    Norm: {grad.norm():.4f}")
        return grad  # Return modified gradient (or original)

    # Register hook
    handle = w.register_hook(hook_fn)

    # Forward + backward
    y = torch.mm(x, w)
    loss = y.sum()

    print("Before backward pass:")
    print(f"  x: {x.shape}, requires_grad={x.requires_grad}")
    print(f"  w: {w.shape}, requires_grad={w.requires_grad}")

    print("\nDuring backward pass:")
    loss.backward()

    print("\nAfter backward pass:")
    print(f"  w.grad: {w.grad.shape}, norm={w.grad.norm():.4f}")

    # Remove hook
    handle.remove()

    print("\nUse cases:")
    print("- Gradient clipping on specific tensors")
    print("- Debugging gradient computation")
    print("- Implementing custom gradient modifications")

    print("=" * 60)
    print()


def gradient_flow_analysis():
    """
    Exercise 4: Analyzing Gradient Flow

    Questions:
    - How do you detect vanishing gradients?
    - How do you detect exploding gradients?
    - What layers typically have gradient problems?
    """
    print("=" * 60)
    print("Exercise 4: Gradient Flow Analysis")
    print("=" * 60)

    # Deep network (prone to vanishing gradients without careful design)
    model = nn.Sequential(
        *[nn.Sequential(nn.Linear(100, 100), nn.Sigmoid()) for _ in range(10)],
        nn.Linear(100, 10)
    )

    # Track gradient norms per layer
    gradient_norms = []

    def track_gradient_norm(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                norm = grad_output[0].norm().item()
                gradient_norms.append((name, norm))
        return hook

    # Register hooks on all layers
    for idx, module in enumerate(model):
        module.register_full_backward_hook(track_gradient_norm(f'layer_{idx}'))

    # Forward + backward
    x = torch.randn(32, 100)
    y_true = torch.randint(0, 10, (32,))
    output = model(x)
    loss = F.cross_entropy(output, y_true)
    loss.backward()

    print("Gradient norms by layer (output → input):")
    print(f"{'Layer':<15} {'Gradient Norm':<20} {'Status'}")
    print("-" * 60)

    for name, norm in gradient_norms[:10]:  # Show first 10
        status = ""
        if norm < 1e-6:
            status = "⚠️  VANISHING"
        elif norm > 100:
            status = "⚠️  EXPLODING"

        print(f"{name:<15} {norm:<20.6f} {status}")

    print("\nObservations:")
    print("- Sigmoid activations can cause vanishing gradients")
    print("- Gradients typically shrink as we go deeper (earlier layers)")
    print("- Solutions: ReLU, BatchNorm, skip connections, better init")

    print("=" * 60)
    print()


def gradient_clipping_hook():
    """
    Exercise 5: Gradient Clipping with Hooks

    Questions:
    - How do you implement gradient clipping?
    - When should you clip gradients?
    - Per-parameter vs global norm clipping?
    """
    print("=" * 60)
    print("Exercise 5: Gradient Clipping with Hooks")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

    max_grad_norm = 1.0
    clipped_count = 0

    def gradient_clipper(threshold):
        """Returns a hook that clips gradients."""
        def hook(grad):
            nonlocal clipped_count
            grad_norm = grad.norm()

            if grad_norm > threshold:
                clipped_count += 1
                # Scale gradient to have norm = threshold
                return grad * (threshold / grad_norm)
            return grad
        return hook

    # Register hooks on all parameters
    handles = []
    for param in model.parameters():
        handle = param.register_hook(gradient_clipper(max_grad_norm))
        handles.append(handle)

    # Simulate training step with large gradients
    x = torch.randn(32, 10)
    y_true = torch.randint(0, 10, (32,))

    # Create scenario with large gradients (multiply weights by large number)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(10.0)  # Artificially create large gradients

    # Forward + backward
    output = model(x)
    loss = F.cross_entropy(output, y_true)

    print("Before backward:")
    print(f"  Loss: {loss.item():.4f}")

    loss.backward()

    print("\nAfter backward (with gradient clipping):")
    print(f"  Number of gradients clipped: {clipped_count}/{len(list(model.parameters()))}")

    # Show gradient norms
    for idx, param in enumerate(model.parameters()):
        if param.grad is not None:
            norm = param.grad.norm().item()
            status = "✓ Clipped" if norm <= max_grad_norm + 1e-6 else "Not clipped"
            print(f"  Param {idx}: grad norm = {norm:.4f} ({status})")

    # Clean up
    for handle in handles:
        handle.remove()

    print("\nAlternative: Use torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)")

    print("=" * 60)
    print()


def feature_map_visualization():
    """
    Exercise 6: Feature Map Extraction for Visualization

    Questions:
    - How do you extract intermediate feature maps?
    - How to visualize what a CNN learns?
    - What's the difference between features and gradients?
    """
    print("=" * 60)
    print("Exercise 6: Feature Map Extraction")
    print("=" * 60)

    # Simple CNN
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
    )

    # Storage for feature maps
    feature_maps = {}

    def save_feature_map(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()
        return hook

    # Register hooks
    model[0].register_forward_hook(save_feature_map('conv1'))
    model[3].register_forward_hook(save_feature_map('conv2'))
    model[6].register_forward_hook(save_feature_map('conv3'))

    # Forward pass with dummy image
    x = torch.randn(1, 3, 64, 64)  # (batch, channels, H, W)
    output = model(x)

    print("Feature maps extracted:")
    for name, fmap in feature_maps.items():
        print(f"\n  {name}:")
        print(f"    Shape: {fmap.shape}")
        print(f"    Channels: {fmap.shape[1]}")
        print(f"    Spatial size: {fmap.shape[2:][0]}x{fmap.shape[3]}")
        print(f"    Mean activation: {fmap.mean():.4f}")
        print(f"    % of neurons active (>0): {(fmap > 0).float().mean():.2%}")

    print("\nUse cases:")
    print("- Visualizing what filters learn at each layer")
    print("- Understanding CNN representations")
    print("- Debugging network architecture")
    print("- Implementing style transfer (Gram matrices)")

    print("=" * 60)
    print()


def main():
    """Run all hook demonstrations."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}\n")

    forward_hooks()
    backward_hooks()
    tensor_hooks()
    gradient_flow_analysis()
    gradient_clipping_hook()
    feature_map_visualization()

    print("\n" + "=" * 60)
    print("PRACTICE QUESTIONS:")
    print("=" * 60)
    print("""
    1. Implement a hook that prints gradient statistics (min, max, mean, std)
       for all layers during training. Identify which layers have smallest gradients.

    2. Create a "gradient monitor" that detects when gradients are vanishing
       (norm < threshold) or exploding (norm > threshold) and sends an alert.

    3. Implement Class Activation Mapping (CAM):
       - Extract final conv layer activations
       - Get gradients w.r.t. class score
       - Compute weighted combination
       - Visualize which regions influence the prediction

    4. Use hooks to implement "gradient checkpointing" manually:
       - Don't store intermediate activations during forward
       - Recompute them during backward pass
       - Compare memory usage with normal training

    5. Build a tool that tracks activation distribution across layers:
       - Mean, std, min, max per layer
       - Plot histograms
       - Detect dead neurons (always 0) or saturated neurons

    6. Implement "discriminative layer freezing":
       - Register hooks that zero out gradients for specific layers
       - Train only the last few layers (transfer learning)
       - Compare with setting requires_grad=False

    7. Create a hook-based profiler:
       - Measure forward/backward time per layer
       - Track memory consumption per layer
       - Identify bottlenecks in your network
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()
