"""
Building Custom Neural Network Modules
=======================================

Learn how to build custom nn.Module classes from scratch.
Understand the fundamentals of PyTorch's neural network building blocks.

Run: python src/custom_modules.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLinearModule(nn.Module):
    """
    Exercise 1: Basic nn.Module Implementation

    Questions:
    - Why do we call super().__init__()?
    - What does nn.Parameter do?
    - Why separate __init__ and forward?
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        # Learnable parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Matrix multiplication: (batch, in_features) @ (in_features, out_features)^T
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        # For pretty printing
        return f'in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}'


class CustomActivation(nn.Module):
    """
    Exercise 2: Custom Activation Function

    Implement: f(x) = x * sigmoid(beta * x) (Swish activation)

    Questions:
    - How do you make activation parameters learnable?
    - What's the difference between functional and module activations?
    """

    def __init__(self, beta=1.0, learnable=False):
        super().__init__()
        if learnable:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('beta', torch.tensor(beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class MultiLayerPerceptron(nn.Module):
    """
    Exercise 3: Multi-Layer Perceptron (MLP)

    Questions:
    - When to use nn.Sequential vs explicit layers?
    - How does dropout work during train vs eval mode?
    - What's the purpose of nn.ModuleList?
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super().__init__()

        # Build layers dynamically
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    """
    Exercise 4: Residual Block (from ResNet)

    Implements: output = ReLU(x + F(x))

    Questions:
    - Why do residual connections help training?
    - When do you need a projection shortcut?
    - How do residual connections affect gradient flow?
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut path (projection if dimensions change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Add shortcut
        out += self.shortcut(identity)
        out = F.relu(out)

        return out


class AttentionModule(nn.Module):
    """
    Exercise 5: Simple Self-Attention

    Implements scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d)V

    Questions:
    - Why scale by √d_k?
    - What's the computational complexity of attention?
    - How does attention differ from convolution?
    """

    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)

        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # scores: (batch, num_heads, seq_len, seq_len)

        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # (batch, num_heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)

        # Final projection
        output = self.out_proj(attn_output)

        return output


def test_modules():
    """Test all custom modules."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    print("=" * 60)
    print("Testing Custom Modules")
    print("=" * 60)

    # Test 1: SimpleLinearModule
    print("\n1. SimpleLinearModule")
    print("-" * 60)
    linear = SimpleLinearModule(10, 5).to(device)
    x = torch.randn(32, 10).to(device)  # batch_size=32
    out = linear(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Module: {linear}")
    print(f"Parameters: {sum(p.numel() for p in linear.parameters())}")

    # Test 2: CustomActivation
    print("\n2. CustomActivation (Swish)")
    print("-" * 60)
    activation = CustomActivation(beta=1.0, learnable=True).to(device)
    x = torch.randn(32, 10).to(device)
    out = activation(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Beta parameter: {activation.beta.item():.4f}")
    print(f"Is beta learnable? {activation.beta.requires_grad}")

    # Test 3: MultiLayerPerceptron
    print("\n3. MultiLayerPerceptron")
    print("-" * 60)
    mlp = MultiLayerPerceptron(input_dim=784, hidden_dims=[512, 256, 128],
                                output_dim=10, dropout=0.2).to(device)
    x = torch.randn(64, 784).to(device)
    out = mlp(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    print(f"\nArchitecture:\n{mlp}")

    # Test 4: ResidualBlock
    print("\n4. ResidualBlock")
    print("-" * 60)
    res_block = ResidualBlock(in_channels=64, out_channels=64).to(device)
    x = torch.randn(16, 64, 32, 32).to(device)  # (batch, channels, height, width)
    out = res_block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in res_block.parameters()):,}")

    # Test with dimension change
    res_block_proj = ResidualBlock(in_channels=64, out_channels=128, stride=2).to(device)
    out_proj = res_block_proj(x)
    print(f"With projection - Output shape: {out_proj.shape}")

    # Test 5: AttentionModule
    print("\n5. AttentionModule (Self-Attention)")
    print("-" * 60)
    attn = AttentionModule(embed_dim=512, num_heads=8).to(device)
    x = torch.randn(32, 10, 512).to(device)  # (batch, seq_len, embed_dim)
    out = attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Number of heads: {attn.num_heads}")
    print(f"Head dimension: {attn.head_dim}")
    print(f"Parameters: {sum(p.numel() for p in attn.parameters()):,}")

    print("\n" + "=" * 60)


def demonstrate_nn_sequential_vs_modulelist():
    """
    Exercise 6: nn.Sequential vs nn.ModuleList

    Question: When should you use each?
    """
    print("\n" + "=" * 60)
    print("nn.Sequential vs nn.ModuleList")
    print("=" * 60)

    # Using nn.Sequential (automatic forward pass)
    seq_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

    print("\nnn.Sequential:")
    print(seq_model)
    x = torch.randn(32, 10)
    out = seq_model(x)  # Automatic forward through all layers
    print(f"Output shape: {out.shape}")

    # Using nn.ModuleList (manual forward pass)
    class ModuleListModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 10)
            ])

        def forward(self, x):
            # Manual iteration required
            for layer in self.layers:
                x = layer(x)
            return x

    modulelist_model = ModuleListModel()

    print("\nnn.ModuleList:")
    print(modulelist_model)
    out = modulelist_model(x)
    print(f"Output shape: {out.shape}")

    print("\nKey Difference:")
    print("- nn.Sequential: Automatic sequential forward pass")
    print("- nn.ModuleList: Manual control (useful for complex logic)")

    print("=" * 60)


def main():
    """Run all demonstrations."""
    test_modules()
    demonstrate_nn_sequential_vs_modulelist()

    print("\n" + "=" * 60)
    print("PRACTICE QUESTIONS:")
    print("=" * 60)
    print("""
    1. Implement a custom layer that performs: y = x * W + b, where W is a
       weight matrix and b is bias. Test it with random inputs.

    2. Build a CNN module with 3 conv layers (each followed by ReLU and MaxPool).
       Input: (batch, 3, 224, 224). Output: (batch, num_classes).

    3. Modify ResidualBlock to use bottleneck architecture (1x1 -> 3x3 -> 1x1
       convolutions). How does this change the parameter count?

    4. Implement a simple Transformer encoder block using AttentionModule,
       LayerNorm, and a feedforward network. Test on sequence data.

    5. Create a module that dynamically changes its architecture based on input
       size. For example, adjust the number of layers based on sequence length.

    6. Implement a custom layer with learnable parameters that applies different
       transformations to even and odd indices of the input.

    7. Build a U-Net-style architecture with skip connections. How do skip
       connections differ from residual connections?
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()
