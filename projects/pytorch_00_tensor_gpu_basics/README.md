# PyTorch Tensor & GPU Basics

Welcome to your first PyTorch practice project! This project focuses on fundamental tensor operations and GPU computing with CUDA.

## Learning Objectives

By completing this project, you will:
- Master PyTorch tensor creation and manipulation
- Understand CPU ↔ GPU data transfer and its overhead
- Learn when GPU acceleration helps vs when it adds overhead
- Monitor and optimize GPU memory usage
- Benchmark operations to measure real performance

## Project Structure

```
pytorch_00_tensor_gpu_basics/
├── src/
│   ├── tensor_basics.py        # Tensor operations, device management
│   ├── gpu_benchmarks.py       # CPU vs GPU performance comparison
│   └── memory_profiling.py     # CUDA memory monitoring
├── data/                       # Generated metrics (git-ignored)
├── params.yaml                 # Configuration parameters
├── dvc.yaml                    # DVC pipeline
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Setup

### 1. Install Dependencies

```bash
cd projects/pytorch_00_tensor_gpu_basics
pip install -r requirements.txt
```

### 2. Verify CUDA Setup

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output on Lambda Labs A100:
```
CUDA available: True
GPU: NVIDIA A100-PCIE-40GB
```

## Exercises

### Exercise 1: Tensor Basics (`tensor_basics.py`)

**Run:**
```bash
python src/tensor_basics.py
```

**What you'll learn:**
- Creating tensors on CPU and GPU
- Device management (`to()`, `.cuda()`, `.cpu()`)
- Measuring CPU ↔ GPU transfer overhead
- Basic tensor operations (addition, matrix multiplication)
- In-place operations (`.add_()` vs `.add()`)
- Indexing, slicing, and boolean masking

**Practice Questions:**
1. Create a 5x5 random tensor on GPU, multiply by 2.5, and move to CPU. Measure total time.
2. Create two 10000×10000 tensors on CPU. Transfer to GPU and do matmul. Compare with CPU matmul.
3. Create tensor [0, 1, 2, ..., 99] on GPU. Use boolean indexing to extract even numbers.
4. Implement in-place normalization: `(x - mean) / std` using `sub_()` and `div_()`.
5. Create a tensor, reshape with `.view()`, modify the view. Explain what happens to original.

**Expected insights:**
- GPU transfers are expensive (milliseconds even for small tensors)
- In-place operations save memory by not creating new tensors
- `.view()` creates a view (shared memory), not a copy

---

### Exercise 2: GPU Benchmarks (`gpu_benchmarks.py`)

**Run:**
```bash
python src/gpu_benchmarks.py
```

**What you'll learn:**
- Benchmarking methodology (warmup, synchronization)
- CPU vs GPU speedup for different operations
- When GPU overhead outweighs benefits (small tensors)
- Impact of batch size on GPU utilization

**Key Benchmarks:**
1. **Matrix Multiplication**: At what size does GPU beat CPU?
2. **Element-wise Ops**: How fast are `sin`, `cos`, `exp` on GPU?
3. **Reductions**: Performance of `sum`, `mean`, `max`
4. **Small Tensor Overhead**: GPU slower for tiny tensors?
5. **Batched Operations**: Larger batches = better GPU utilization

**Practice Questions:**
1. Run benchmarks on A100. At what matrix size does GPU become faster than CPU?
2. Benchmark `torch.bmm()` vs `torch.mm()` in a loop for batch processing.
3. Implement convolution benchmark (`F.conv2d`). Test various image/batch sizes.
4. Add CPU→GPU transfer time to benchmarks. How much does it affect total time?
5. Research CUDA streams. Benchmark overlapping transfer with computation.

**Expected insights:**
- GPU excels at large matrix operations (1024×1024+)
- Small tensors (<64×64) may be faster on CPU due to kernel launch overhead
- Batch size significantly affects GPU throughput
- Element-wise operations show 5-20x speedup on GPU

---

### Exercise 3: Memory Profiling (`memory_profiling.py`)

**Run:**
```bash
python src/memory_profiling.py
```

**What you'll learn:**
- GPU memory allocation vs reservation (caching allocator)
- Calculating tensor memory footprint
- Memory savings from in-place operations
- Gradient memory overhead (autograd)
- Inference optimization with `torch.no_grad()` and `torch.inference_mode()`

**Key Concepts:**
- **Allocated**: Memory actively used by tensors
- **Reserved**: Memory cached by PyTorch allocator
- **float32**: 4 bytes/element, **float16**: 2 bytes/element
- Gradients double memory usage during training

**Practice Questions:**
1. Create a function to monitor GPU memory during a training loop (print every N iterations).
2. Implement manual gradient checkpointing: forward pass in segments, store minimal activations.
3. Compare float32 vs float16 training memory. How much do you save?
4. Profile ResNet50 memory usage. Which layers use the most memory?
5. Implement gradient accumulation to simulate large batch size with small batches.

**Expected insights:**
- `torch.cuda.empty_cache()` doesn't free memory immediately (caching)
- In-place ops save ~2x memory (no temporary tensors)
- `torch.no_grad()` critical for inference (no gradient storage)
- float16 saves 50% memory vs float32 (critical for large models)

---

## Running with DVC

Track experiments with DVC:

```bash
# Run all stages
dvc repro

# Run specific stage
dvc repro tensor_basics

# View metrics
dvc metrics show
```

## Advanced Challenges

### Challenge 1: Custom CUDA Kernel (Advanced)
Research PyTorch's CUDA extension interface. Implement a simple element-wise operation (e.g., `y = x^2 + 2x + 1`) as a CUDA kernel. Compare performance with native PyTorch.

### Challenge 2: Multi-GPU Data Parallelism
If you have access to multiple GPUs, implement data parallelism:
- Split a batch across GPUs
- Run forward pass in parallel
- Gather results

Compare with `torch.nn.DataParallel`.

### Challenge 3: Mixed Precision Training
Implement a training loop using `torch.cuda.amp.autocast()` and `GradScaler`. Measure:
- Memory savings
- Training speed improvement
- Final model accuracy

### Challenge 4: Profiling with PyTorch Profiler
Use `torch.profiler` to profile a training loop:
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
) as prof:
    # Your training code
print(prof.key_averages().table())
```

Identify bottlenecks (data loading, forward, backward, optimizer step).

## Key Takeaways

✅ **GPU is not always faster**: Small tensors have kernel launch overhead
✅ **Minimize CPU↔GPU transfers**: Keep data on GPU during training
✅ **Batch operations**: Larger batches better utilize GPU parallelism
✅ **In-place operations**: Save memory (critical for large models)
✅ **Use `torch.no_grad()` for inference**: Avoid storing gradients
✅ **Monitor memory**: Use `torch.cuda.memory_allocated()` to debug OOM errors
✅ **Mixed precision (float16)**: 2x memory savings, faster training on modern GPUs
✅ **Always call `torch.cuda.synchronize()`**: When benchmarking GPU operations

## Next Steps

Once you've completed all exercises, move to:
- **Project 1**: `pytorch_01_neural_networks` - Building custom modules and layers
- **Project 2**: `pytorch_02_training_optimization` - Complete training loops with mixed precision
- **Project 3**: `pytorch_03_computer_vision` - CNNs and real-world image classification

## Resources

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size in `params.yaml`
- Use `torch.cuda.empty_cache()` between experiments
- Monitor memory with `memory_profiling.py`

### "CUDA not available"
- Check PyTorch installation: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Verify NVIDIA driver: `nvidia-smi`

### Slow GPU performance
- Check GPU utilization: `nvidia-smi dmon`
- Increase batch size (better GPU utilization)
- Ensure data is on GPU (not transferring every iteration)

---

**Happy Learning!** 🚀

Master these fundamentals and you'll be ready to tackle deep learning at scale on your A100 GPU.
