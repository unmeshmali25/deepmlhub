"""
PyTorch Tensor Basics and GPU Operations
=========================================

This module covers fundamental tensor operations and GPU device management.
Focus on understanding tensor creation, device transfers, and basic operations.

Key Topics Covered:
-------------------
- CUDA availability checking and GPU information
- Tensor creation on CPU vs GPU devices
- Measuring CPU <-> GPU transfer overhead
- Basic tensor operations (addition, matrix multiplication, broadcasting)
- In-place operations for memory efficiency
- Tensor indexing, slicing, and advanced indexing
- Device mismatch handling and error management

Learning Objectives:
--------------------
1. Understand the difference between CPU and GPU tensor storage
2. Learn efficient device transfer strategies to minimize overhead
3. Master in-place operations for memory-constrained environments
4. Practice tensor manipulation techniques (indexing, slicing, views)
5. Recognize and handle device mismatch errors

Prerequisites:
--------------
- PyTorch installed with CUDA support (optional but recommended)
- Basic understanding of Python and NumPy arrays
- Familiarity with matrix operations

Usage:
------
    python src/tensor_basics.py

The script will automatically detect CUDA availability and adjust exercises
accordingly. If CUDA is not available, operations will run on CPU with
appropriate warnings.

Performance Notes:
------------------
- CPU-GPU transfers are expensive; minimize them in production code
- Keep tensors on GPU for sequences of operations
- Use in-place operations (methods ending with '_') to save memory
- Synchronize with torch.cuda.synchronize() for accurate timing

Author: Unmesh Mali
Date: 2026
"""

import torch
import time
import sys


def check_cuda_availability():
    """Check CUDA availability and print GPU information."""
    print("=" * 60)
    print("CUDA Availability Check")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("WARNING: CUDA not available. Code will run on CPU.")
    print("=" * 60)
    print()


def tensor_creation_basics():
    """
    Exercise 1: Tensor Creation on Different Devices

    Questions to explore:
    - What's the default device for tensor creation?
    - How do you explicitly create tensors on GPU?
    - What's the difference between .cuda() and .to('cuda')?
    """
    print("Exercise 1: Tensor Creation")
    print("-" * 60)

    # Create tensor on CPU (default)
    cpu_tensor = torch.randn(3, 3)
    print(f"CPU Tensor:\n{cpu_tensor}")
    print(f"Device: {cpu_tensor.device}")
    print(f"Data type: {cpu_tensor.dtype}")
    print()

    # Create tensor directly on GPU
    if torch.cuda.is_available():
        gpu_tensor = torch.randn(3, 3, device="cuda")
        print(f"GPU Tensor:\n{gpu_tensor}")
        print(f"Device: {gpu_tensor.device}")

        # Alternative: Create on CPU then move to GPU
        cpu_to_gpu = torch.randn(3, 3).cuda()
        print(f"\nCPU->GPU Tensor device: {cpu_to_gpu.device}")

        # Using .to() method (more flexible)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        flexible_tensor = torch.randn(3, 3).to(device)
        print(f"Flexible tensor device: {flexible_tensor.device}")

    print("=" * 60)
    print()


def measure_device_transfer_overhead():
    """
    Exercise 2: Measure CPU <-> GPU Transfer Overhead

    Questions to explore:
    - How expensive is CPU-GPU data transfer?
    - Does transfer time scale linearly with tensor size?
    - Should you batch multiple small transfers or do them individually?
    """
    print("Exercise 2: Device Transfer Overhead")
    print("-" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping this exercise.")
        return

    sizes = [100, 1000, 10000, 100000]

    print(f"{'Size':<15} {'CPU->GPU (ms)':<20} {'GPU->CPU (ms)':<20}")
    print("-" * 60)

    for size in sizes:
        # Create tensor on CPU
        cpu_tensor = torch.randn(size, size)

        # Measure CPU -> GPU transfer
        start = time.time()
        gpu_tensor = cpu_tensor.cuda()
        torch.cuda.synchronize()  # Wait for GPU operation to complete
        cpu_to_gpu_time = (time.time() - start) * 1000

        # Measure GPU -> CPU transfer
        start = time.time()
        back_to_cpu = gpu_tensor.cpu()
        gpu_to_cpu_time = (time.time() - start) * 1000

        print(f"{size:<15} {cpu_to_gpu_time:<20.4f} {gpu_to_cpu_time:<20.4f}")

    print("\nKey Takeaway: Minimize CPU-GPU transfers! Keep data on GPU when possible.")
    print("=" * 60)
    print()


def tensor_operations_gpu():
    """
    Exercise 3: Basic Tensor Operations on GPU

    Questions to explore:
    - Do operations require tensors to be on the same device?
    - What happens if you mix CPU and GPU tensors?
    - How does broadcasting work on GPU?
    """
    print("Exercise 3: Tensor Operations on GPU")
    print("-" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        device = "cpu"
    else:
        device = "cuda"

    # Create tensors on GPU
    a = torch.randn(3, 3, device=device)
    b = torch.randn(3, 3, device=device)

    # Basic operations
    print("Addition:")
    c = a + b
    print(f"Result device: {c.device}")

    print("\nMatrix Multiplication:")
    d = torch.mm(a, b)
    print(f"Result device: {d.device}")
    print(f"Result:\n{d}")

    # Broadcasting
    print("\nBroadcasting (tensor + scalar):")
    e = a + 5.0
    print(f"Result shape: {e.shape}, device: {e.device}")

    # Demonstrate device mismatch error
    if torch.cuda.is_available():
        print("\n[Demo] Attempting CPU + GPU operation (will fail):")
        try:
            cpu_tensor = torch.randn(3, 3)
            gpu_tensor = torch.randn(3, 3, device="cuda")
            result = cpu_tensor + gpu_tensor  # This will raise an error
        except RuntimeError as e:
            print(f"Error: {e}")
            print("Lesson: All tensors in an operation must be on the same device!")

    print("=" * 60)
    print()


def in_place_operations():
    """
    Exercise 4: In-place Operations and Memory Efficiency

    Questions to explore:
    - What's the difference between a.add(b) and a.add_(b)?
    - When should you use in-place operations?
    - How do in-place ops affect memory usage?
    """
    print("Exercise 4: In-place Operations")
    print("-" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Regular operation (creates new tensor)
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)

    print("Regular operation (creates new tensor):")
    a_id_before = id(a)
    c = a.add(b)
    print(f"Original tensor id: {a_id_before}")
    print(f"Original tensor id after op: {id(a)} (unchanged)")
    print(f"Result tensor id: {id(c)} (new tensor)")

    # In-place operation (modifies existing tensor)
    print("\nIn-place operation (modifies existing tensor):")
    a = torch.randn(1000, 1000, device=device)
    a_id_before = id(a)
    a.add_(b)  # Note the underscore
    print(f"Original tensor id: {a_id_before}")
    print(f"Original tensor id after op: {id(a)} (same!)")
    print(
        "\nIn-place operations (with _ suffix) save memory but modify the original tensor."
    )

    print("=" * 60)
    print()


def tensor_indexing_and_slicing():
    """
    Exercise 5: Indexing, Slicing, and Advanced Indexing

    Questions to explore:
    - How does slicing work on GPU tensors?
    - What's the difference between view() and clone()?
    - How does advanced indexing (boolean masks) work?
    """
    print("Exercise 5: Indexing and Slicing")
    print("-" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create tensor
    x = torch.arange(20, device=device).reshape(4, 5)
    print(f"Original tensor:\n{x}\n")

    # Basic indexing
    print(f"First row: {x[0]}")
    print(f"First column: {x[:, 0]}")
    print(f"2x2 submatrix: {x[1:3, 2:4]}\n")

    # Boolean masking
    mask = x > 10
    print(f"Boolean mask (x > 10):\n{mask}\n")
    print(f"Elements > 10: {x[mask]}\n")

    # View vs Clone
    y = x.view(20)  # Shared memory
    z = x.clone()  # New memory

    print(f"View shares memory: {y.data_ptr() == x.data_ptr()}")
    print(f"Clone creates new memory: {z.data_ptr() == x.data_ptr()}")

    print("=" * 60)
    print()


def main():
    """Run all exercises."""
    check_cuda_availability()
    tensor_creation_basics()
    measure_device_transfer_overhead()
    tensor_operations_gpu()
    in_place_operations()
    tensor_indexing_and_slicing()

    print("\n" + "=" * 60)
    print("PRACTICE QUESTIONS:")
    print("=" * 60)
    print("""
    1. Create a 5x5 random tensor on GPU, multiply it by 2.5, and move it back to CPU.
       Measure the total time including transfers.

    2. Create two large tensors (10000x10000) on CPU. Transfer them to GPU and
       perform matrix multiplication. Compare with CPU matmul time.

    3. Create a tensor with values [0, 1, 2, ..., 99]. Use boolean indexing to
       extract all even numbers. Do this on GPU.

    4. Implement a function that takes a tensor and normalizes it to mean=0, std=1
       in-place (hint: use tensor.sub_() and tensor.div_()).

    5. Create a tensor, reshape it using .view(), modify the view, and observe
       how it affects the original tensor. Explain why.
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()
