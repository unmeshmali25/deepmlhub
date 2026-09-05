import torch
import gc


def print_gpu_memory():
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9

    print(f"  Allocated: {allocated:.3f} GB")
    print(f"  Reserved:  {reserved:.3f} GB")
    print(f"  Max Allocated: {max_allocated:.3f} GB")


def memory_basics():
    """
    Exercise 1: Understanding GPU Memory Allocation

    Questions:
    - What's the difference between allocated and reserved memory?
    - How does PyTorch's caching allocator work?
    - When is memory actually freed?
    """
    print("=" * 60)
    print("Exercise 1: GPU Memory Basics")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print("\n1. Initial state (after cache clear):")
    print_gpu_memory()

    print("\n2. Creating a 1GB tensor:")
    x = torch.randn(
        256, 1024, 1024, device="cuda"
    )  # ~1GB (256 * 1024 * 1024 * 4 bytes)
    print_gpu_memory()

    print("\n3. Creating another 1GB tensor:")
    y = torch.randn(256, 1024, 1024, device="cuda")
    print_gpu_memory()

    print("\n4. Deleting first tensor (del x):")
    del x
    print_gpu_memory()
    print("  Note: Memory still reserved by PyTorch caching allocator!")

    print("\n5. Emptying cache (torch.cuda.empty_cache()):")
    torch.cuda.empty_cache()
    print_gpu_memory()
    print("  Note: Now memory is actually freed.")

    # Cleanup
    del y
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print()


def tensor_size_and_memory():
    """
    Exercise 2: Calculate Memory Usage of Tensors

    Questions:
    - How do you calculate tensor memory size?
    - What's the memory cost of different dtypes?
    - How does memory scale with batch size?
    """
    print("=" * 60)
    print("Exercise 2: Tensor Size and Memory Calculation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtypes = [torch.float32, torch.float16, torch.int64, torch.int32, torch.int8]

    print(f"{'Data Type':<15} {'Bytes/Element':<20} {'Memory for (1000,1000)':<25}")
    print("-" * 60)

    for dtype in dtypes:
        x = torch.randn(1000, 1000, dtype=dtype, device=device)
        bytes_per_element = x.element_size()
        total_bytes = x.numel() * bytes_per_element
        total_mb = total_bytes / 1e6

        print(f"{str(dtype):<15} {bytes_per_element:<20} {total_mb:<25.2f} MB")

        if device == "cuda":
            del x
            torch.cuda.empty_cache()

    print("\nKey Insight: float16 (half precision) uses 50% less memory than float32!")
    print("Useful for training large models with mixed precision.")

    # Batch size scaling
    print("\n" + "-" * 60)
    print(
        "Memory Scaling with Batch Size (float32, image-like tensor [B, 3, 224, 224]):"
    )
    print("-" * 60)

    batch_sizes = [1, 8, 16, 32, 64, 128, 256]
    channels, height, width = 3, 224, 224

    print(f"{'Batch Size':<15} {'Memory (MB)':<20}")
    print("-" * 60)

    for batch_size in batch_sizes:
        num_elements = batch_size * channels * height * width
        bytes_total = num_elements * 4  # float32 = 4 bytes
        mb = bytes_total / 1e6
        print(f"{batch_size:<15} {mb:<20.2f}")

    print("=" * 60)
    print()


def in_place_memory_savings():
    """
    Exercise 3: In-place Operations for Memory Efficiency

    Questions:
    - How much memory do in-place operations save?
    - When should you use them?
    - What are the risks?
    """
    print("=" * 60)
    print("Exercise 3: In-place Operations and Memory Savings")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    size = (512, 1024, 1024)  # ~2GB tensor

    # Method 1: Regular operations (creates new tensors)
    print("\nMethod 1: Regular operations (x = x + 1)")
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(size, device="cuda")
    initial_mem = torch.cuda.memory_allocated() / 1e9

    x = x + 1.0  # Creates temporary tensor
    x = x * 2.0  # Creates another temporary tensor
    x = torch.relu(x)  # Creates another temporary tensor

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"  Initial memory: {initial_mem:.3f} GB")
    print(f"  Peak memory: {peak_mem:.3f} GB")
    print(f"  Extra memory used: {peak_mem - initial_mem:.3f} GB")

    del x
    torch.cuda.empty_cache()

    # Method 2: In-place operations
    print("\nMethod 2: In-place operations (x.add_(1))")
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(size, device="cuda")
    initial_mem = torch.cuda.memory_allocated() / 1e9

    x.add_(1.0)  # In-place
    x.mul_(2.0)  # In-place
    x.relu_()  # In-place

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"  Initial memory: {initial_mem:.3f} GB")
    print(f"  Peak memory: {peak_mem:.3f} GB")
    print(f"  Extra memory used: {peak_mem - initial_mem:.3f} GB")

    del x
    torch.cuda.empty_cache()

    print("\nKey Insight: In-place operations avoid creating temporary tensors,")
    print("saving GPU memory. Critical for training large models!")

    print("=" * 60)
    print()


def gradient_memory_overhead():
    """
    Exercise 4: Memory Overhead from Autograd

    Questions:
    - How much memory do gradients use?
    - What does .backward() store?
    - How does torch.no_grad() save memory?
    """
    print("=" * 60)
    print("Exercise 4: Gradient Memory Overhead")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch.cuda.empty_cache()
    size = (256, 1024, 1024)  # ~1GB tensor

    # With gradient tracking
    print("\n1. With gradient tracking (requires_grad=True):")
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(size, device="cuda", requires_grad=True)
    y = x * 2
    z = y.mean()

    mem_before_backward = torch.cuda.memory_allocated() / 1e9
    print(f"  Memory before backward: {mem_before_backward:.3f} GB")

    z.backward()

    mem_after_backward = torch.cuda.memory_allocated() / 1e9
    print(f"  Memory after backward: {mem_after_backward:.3f} GB")
    print(f"  Gradient memory: {mem_after_backward - mem_before_backward:.3f} GB")

    if x.grad is not None:
        print(f"  x.grad size: {x.grad.numel() * x.grad.element_size() / 1e9:.3f} GB")

    del x, y, z
    torch.cuda.empty_cache()

    # Without gradient tracking
    print("\n2. Without gradient tracking (torch.no_grad()):")
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        x = torch.randn(size, device="cuda")
        y = x * 2
        z = y.mean()

    mem_no_grad = torch.cuda.memory_allocated() / 1e9
    print(f"  Memory used: {mem_no_grad:.3f} GB")
    print(f"  No gradients stored!")

    del x, y, z
    torch.cuda.empty_cache()

    print("\nKey Insight: Use torch.no_grad() or torch.inference_mode() during")
    print("inference to save memory by not storing gradients!")

    print("=" * 60)
    print()


def memory_efficient_inference():
    """
    Exercise 5: Memory-Efficient Inference Patterns

    Questions:
    - How does batch size affect memory during inference?
    - What's the benefit of torch.inference_mode()?
    - How can you process large datasets with limited memory?
    """
    print("=" * 60)
    print("Exercise 5: Memory-Efficient Inference")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    # Simulate a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 10),
    ).cuda()

    batch_sizes = [1, 16, 64, 256, 1024]

    print("\nMemory usage for different batch sizes during inference:")
    print(f"{'Batch Size':<15} {'Memory (GB)':<20}")
    print("-" * 60)

    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():  # More efficient than torch.no_grad()
            x = torch.randn(batch_size, 1024, device="cuda")
            y = model(x)

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"{batch_size:<15} {peak_mem:<20.3f}")

    print("\nBest Practice: For inference on large datasets, use smaller batches")
    print("with torch.inference_mode() to minimize memory usage.")

    print("=" * 60)
    print()


def main():
    """Run all memory profiling exercises."""
    print("\n" + "=" * 60)
    print("CUDA MEMORY PROFILING GUIDE")
    print("=" * 60)

    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {total_memory:.2f} GB")
    else:
        print("⚠️  WARNING: CUDA not available.")

    print("=" * 60)
    print()

    memory_basics()
    tensor_size_and_memory()
    in_place_memory_savings()
    gradient_memory_overhead()
    memory_efficient_inference()

    print("\n" + "=" * 60)
    print("PRACTICE QUESTIONS:")
    print("=" * 60)
    print("""
    1. Create a function that monitors GPU memory usage in real-time during
       a training loop. Print memory stats every N iterations.

    2. Implement gradient checkpointing manually: compute forward pass in
       segments, only storing intermediate activations needed for backward.

    3. Compare memory usage of training a model with float32 vs float16.
       How much memory do you save?

    4. Profile memory usage of a large model (e.g., ResNet50). Identify which
       layers consume the most memory.

    5. Research and implement gradient accumulation: simulate a large batch
       size by accumulating gradients over multiple small batches. How does
       this affect memory usage?
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()
