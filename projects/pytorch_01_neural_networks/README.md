# PyTorch Neural Networks

Deep dive into building custom neural network modules, understanding layers, initialization, and gradient flow.

## Learning Objectives

By completing this project, you will:
- Build custom `nn.Module` classes from scratch
- Understand PyTorch's layer types (Linear, Conv, BatchNorm, etc.)
- Master weight initialization strategies (Xavier, Kaiming, Orthogonal)
- Use hooks to inspect activations and gradients
- Debug gradient flow and detect vanishing/exploding gradients

## Prerequisites

Complete **Project 0: Tensor & GPU Basics** first for foundational knowledge.

## Project Structure

```
pytorch_01_neural_networks/
├── src/
│   ├── custom_modules.py          # Build nn.Module from scratch
│   ├── layer_playground.py        # Explore different layer types
│   ├── initialization_study.py    # Weight initialization strategies
│   └── hooks_and_gradients.py     # Inspect activations/gradients
├── params.yaml                    # Configuration
├── dvc.yaml                       # DVC pipeline
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Setup

```bash
cd projects/pytorch_01_neural_networks
pip install -r requirements.txt
```

## Exercises

### Exercise 1: Custom Modules (`custom_modules.py`)

**Run:**
```bash
python src/custom_modules.py
```

**What you'll build:**
1. **SimpleLinearModule**: Basic `nn.Module` with learnable parameters
2. **CustomActivation**: Swish activation with learnable β parameter
3. **MultiLayerPerceptron**: Dynamic MLP with configurable hidden layers
4. **ResidualBlock**: Residual connection (like ResNet)
5. **AttentionModule**: Self-attention mechanism (Transformer building block)

**Key Concepts:**
- `super().__init__()` - Initialize parent class
- `nn.Parameter()` - Make tensors learnable
- `register_buffer()` - Store non-learnable tensors (e.g., running stats)
- `forward()` - Define computation graph
- `nn.Sequential` vs `nn.ModuleList` - When to use each

**Practice Questions:**
1. Implement custom layer: `y = x * W + b` with learnable W, b
2. Build CNN with 3 conv layers (each with ReLU + MaxPool)
3. Modify ResidualBlock to use bottleneck architecture (1x1 → 3x3 → 1x1)
4. Implement Transformer encoder block (Attention + FFN + LayerNorm)
5. Create module with dynamic architecture based on input size
6. Build U-Net architecture with skip connections

**Expected insights:**
- `nn.Module` is the foundation of all PyTorch models
- Parameters auto-register when assigned as attributes
- `forward()` defines the computation, PyTorch handles backward automatically

---

### Exercise 2: Layer Playground (`layer_playground.py`)

**Run:**
```bash
python src/layer_playground.py
```

**Layers covered:**
1. **Linear**: Fully connected layers, parameter count
2. **Conv2d**: Convolutional layers, output size calculation
3. **BatchNorm/LayerNorm/GroupNorm**: Different normalization strategies
4. **MaxPool/AvgPool/AdaptiveAvgPool**: Pooling operations
5. **Dropout/Dropout2d**: Regularization
6. **Activations**: ReLU, LeakyReLU, ELU, GELU, Sigmoid, Tanh

**Key Formulas:**
- **Linear params**: `(in_features * out_features) + out_features`
- **Conv2d params**: `(in_channels * out_channels * kernel_h * kernel_w) + out_channels`
- **Conv output size**: `(H + 2P - K) / S + 1`

**Practice Questions:**
1. Calculate output size for Conv2d(3, 64, kernel_size=5, stride=2, padding=2) with input (8, 3, 128, 128)
2. Build Conv → BatchNorm → ReLU → MaxPool module. Test train vs eval mode for BN
3. Implement depthwise separable convolution (MobileNet style)
4. Compare BatchNorm vs LayerNorm vs GroupNorm on residual network
5. Test dropout with different p values (0.1, 0.5, 0.9)
6. Implement Squeeze-and-Excitation (SE) block

**Expected insights:**
- BatchNorm behavior differs in train vs eval mode
- AdaptiveAvgPool enables variable input sizes
- Dropout2d drops entire channels (not pixels)
- GELU is preferred in modern Transformers (vs ReLU)

---

### Exercise 3: Initialization Study (`initialization_study.py`)

**Run:**
```bash
python src/initialization_study.py
```

**Initialization methods:**
1. **Xavier (Glorot)**: For tanh/sigmoid activations
2. **Kaiming (He)**: For ReLU activations
3. **Orthogonal**: For RNNs and very deep networks
4. **Custom**: Different init per layer type

**Key Concepts:**
- **Variance preservation**: Keep activation variance constant across layers
- **Symmetry breaking**: Why zero init doesn't work
- **Scale matters**: Too small → vanishing, too large → exploding

**Practice Questions:**
1. Implement auto-init function: ReLU → Kaiming, Tanh → Xavier
2. Build 50-layer network. Compare default vs Xavier vs Orthogonal convergence
3. Implement LSUV initialization (Layer-Sequential Unit-Variance)
4. Research Fixup initialization for ResNets (no BatchNorm)
5. Design initialization for skip connection networks
6. Experiment with sparse initialization (only some weights non-zero)
7. Research Transformer initialization (attention, FFN, LayerNorm)

**Expected insights:**
- Bad init → exploding/vanishing activations → training failure
- Xavier preserves variance for linear activations
- Kaiming accounts for ReLU zeroing half the values
- Orthogonal init helps gradient flow in RNNs

---

### Exercise 4: Hooks and Gradients (`hooks_and_gradients.py`)

**Run:**
```bash
python src/hooks_and_gradients.py
```

**Hook types:**
1. **Forward hooks**: Capture intermediate activations
2. **Backward hooks**: Inspect gradients during backprop
3. **Tensor hooks**: Register on specific tensors
4. **Gradient clipping hooks**: Prevent exploding gradients

**Use cases:**
- Extract feature maps for visualization
- Detect vanishing/exploding gradients
- Implement custom gradient modifications
- Debug training issues
- Class Activation Mapping (CAM)

**Practice Questions:**
1. Print gradient statistics (min, max, mean, std) for all layers during training
2. Create gradient monitor that alerts on vanishing/exploding gradients
3. Implement Class Activation Mapping (CAM) for CNNs
4. Use hooks to implement gradient checkpointing manually
5. Track activation distribution: mean, std, % dead neurons
6. Implement discriminative layer freezing with hooks (transfer learning)
7. Build hook-based profiler (time + memory per layer)

**Expected insights:**
- Hooks enable inspection without modifying model code
- Forward hooks capture activations, backward hooks capture gradients
- Sigmoid/Tanh cause vanishing gradients in deep networks
- Gradient clipping prevents exploding gradients (critical for RNNs)

---

## Running with DVC

```bash
# Run all stages
dvc repro

# Run specific stage
dvc repro custom_modules

# View generated plots
ls *.png
```

## Advanced Challenges

### Challenge 1: Build Vision Transformer (ViT)
Implement a Vision Transformer from scratch:
- Patch embedding layer
- Multi-head self-attention blocks
- MLP blocks with GELU activation
- Classification head

Compare with CNN on image classification.

### Challenge 2: Implement Gradient Checkpointing
Manually implement gradient checkpointing:
- Discard activations during forward pass
- Recompute them during backward pass
- Measure memory savings vs training time tradeoff

### Challenge 3: Custom Normalization Layer
Implement a custom normalization layer:
- Compute statistics (mean, variance)
- Normalize inputs
- Apply learnable affine transform (γ, β)
- Test on deep network

### Challenge 4: Network Surgery with Hooks
Use hooks to:
- Extract intermediate layer from pretrained model
- Replace it with different architecture
- Fine-tune modified model
- Compare accuracy

### Challenge 5: Activation Atlas
Build an "activation atlas":
- Extract activations from all layers on dataset
- Apply dimensionality reduction (PCA/UMAP)
- Visualize what different layers learn
- Identify "concept neurons"

## Common Pitfalls

### Issue 1: "RuntimeError: grad can be implicitly created only for scalar outputs"
**Solution**: Use `.backward()` only on scalar tensors (usually loss). For non-scalar, use `.backward(gradient)`.

### Issue 2: "Expected all tensors to be on the same device"
**Solution**: Move model and data to same device:
```python
model = model.to(device)
x = x.to(device)
```

### Issue 3: BatchNorm behaving strangely during eval
**Solution**: Always call `model.eval()` before inference:
```python
model.eval()
with torch.no_grad():
    output = model(x)
```

### Issue 4: Vanishing gradients in deep network
**Solutions**:
- Use ReLU instead of Sigmoid/Tanh
- Add BatchNorm after each layer
- Use residual connections (skip connections)
- Use proper weight initialization (Kaiming for ReLU)

### Issue 5: Hook not being called
**Solution**: Keep handle reference alive:
```python
handle = layer.register_forward_hook(hook_fn)
# Don't let handle get garbage collected
```

## Key Takeaways

✅ **nn.Module is the building block**: All PyTorch models inherit from `nn.Module`
✅ **Layers are modules too**: `nn.Linear`, `nn.Conv2d` are all `nn.Module` subclasses
✅ **Initialization matters**: Bad init = training failure (vanishing/exploding)
✅ **Xavier for linear, Kaiming for ReLU**: Match init to activation function
✅ **BatchNorm has train/eval modes**: Always call `.eval()` before inference
✅ **Hooks enable introspection**: Access activations/gradients without modifying code
✅ **Gradient clipping prevents explosion**: Critical for RNNs and very deep networks

## Next Steps

Once you've mastered these concepts, proceed to:
- **Project 2**: `pytorch_02_training_optimization` - Complete training loops, optimizers, mixed precision
- **Project 3**: `pytorch_03_computer_vision` - Apply these concepts to real CV tasks

## Resources

- [PyTorch nn.Module docs](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [Weight Initialization Paper (Glorot & Bengio)](http://proceedings.mlr.press/v9/glorot10a.html)
- [Kaiming Initialization (He et al.)](https://arxiv.org/abs/1502.01852)
- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)

---

**Happy Learning!** 🚀

Master these building blocks and you'll be able to implement any neural network architecture.
