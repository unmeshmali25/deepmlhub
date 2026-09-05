"""
GPU Performance Benchmarking
=============================

Compare CPU vs GPU performance for various operations.
Understand when GPU acceleration provides benefits and when overhead dominates.

Run: python src/gpu_benchmarks.py
"""

import torch
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt


def benchmark_operation(func, device, warmup=5, iterations=20):
    """
    Benchmark a function on a specific device.

    Args:
        func: Function to benchmark
        device: 'cpu' or 'cuda'
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        Average time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        func(device)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.time()
        func(device)
        if device == 'cuda':
            torch.cuda.synchronize()  # Critical: wait for GPU to finish
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)

    return np.mean(times), np.std(times)


def matrix_multiplication_benchmark():
    """
    Benchmark: Matrix Multiplication

    Question: At what matrix size does GPU become faster than CPU?
    """
    print("=" * 60)
    print("Benchmark 1: Matrix Multiplication (A @ B)")
    print("=" * 60)

    sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    cpu_times = []
    gpu_times = []

    print(f"{'Size':<10} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for size in sizes:
        # Define operation
        def matmul_op(device):
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            c = torch.mm(a, b)
            return c

        # Benchmark CPU
        cpu_mean, cpu_std = benchmark_operation(matmul_op, 'cpu', warmup=3, iterations=10)
        cpu_times.append(cpu_mean)

        # Benchmark GPU
        if torch.cuda.is_available():
            gpu_mean, gpu_std = benchmark_operation(matmul_op, 'cuda', warmup=3, iterations=10)
            gpu_times.append(gpu_mean)
            speedup = cpu_mean / gpu_mean
            print(f"{size:<10} {cpu_mean:<15.4f} {gpu_mean:<15.4f} {speedup:<10.2f}x")
        else:
            gpu_times.append(0)
            print(f"{size:<10} {cpu_mean:<15.4f} {'N/A':<15} {'N/A':<10}")

    # Plot results
    if torch.cuda.is_available():
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, cpu_times, 'o-', label='CPU', linewidth=2)
        plt.plot(sizes, gpu_times, 's-', label='GPU (CUDA)', linewidth=2)
        plt.xlabel('Matrix Size (N x N)')
        plt.ylabel('Time (ms)')
        plt.title('Matrix Multiplication: CPU vs GPU')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('projects/pytorch_00_tensor_gpu_basics/matmul_benchmark.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: matmul_benchmark.png")

    print("=" * 60)
    print()


def element_wise_operations_benchmark():
    """
    Benchmark: Element-wise Operations

    Question: Are element-wise ops faster on GPU?
    """
    print("=" * 60)
    print("Benchmark 2: Element-wise Operations (sin, cos, exp)")
    print("=" * 60)

    size = 10_000_000  # 10M elements

    operations = {
        'sin': lambda x: torch.sin(x),
        'cos': lambda x: torch.cos(x),
        'exp': lambda x: torch.exp(x),
        'sqrt': lambda x: torch.sqrt(torch.abs(x)),
        'tanh': lambda x: torch.tanh(x),
    }

    print(f"{'Operation':<15} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for op_name, op_func in operations.items():
        def operation(device):
            x = torch.randn(size, device=device)
            return op_func(x)

        cpu_mean, _ = benchmark_operation(operation, 'cpu', warmup=3, iterations=10)

        if torch.cuda.is_available():
            gpu_mean, _ = benchmark_operation(operation, 'cuda', warmup=3, iterations=10)
            speedup = cpu_mean / gpu_mean
            print(f"{op_name:<15} {cpu_mean:<15.4f} {gpu_mean:<15.4f} {speedup:<10.2f}x")
        else:
            print(f"{op_name:<15} {cpu_mean:<15.4f} {'N/A':<15} {'N/A':<10}")

    print("=" * 60)
    print()


def reduction_operations_benchmark():
    """
    Benchmark: Reduction Operations (sum, mean, max)

    Question: How do reductions perform on GPU vs CPU?
    """
    print("=" * 60)
    print("Benchmark 3: Reduction Operations")
    print("=" * 60)

    size = 100_000_000  # 100M elements

    operations = {
        'sum': lambda x: torch.sum(x),
        'mean': lambda x: torch.mean(x),
        'max': lambda x: torch.max(x),
        'min': lambda x: torch.min(x),
        'std': lambda x: torch.std(x),
    }

    print(f"{'Operation':<15} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for op_name, op_func in operations.items():
        def operation(device):
            x = torch.randn(size, device=device)
            return op_func(x)

        cpu_mean, _ = benchmark_operation(operation, 'cpu', warmup=3, iterations=10)

        if torch.cuda.is_available():
            gpu_mean, _ = benchmark_operation(operation, 'cuda', warmup=3, iterations=10)
            speedup = cpu_mean / gpu_mean
            print(f"{op_name:<15} {cpu_mean:<15.4f} {gpu_mean:<15.4f} {speedup:<10.2f}x")
        else:
            print(f"{op_name:<15} {cpu_mean:<15.4f} {'N/A':<15} {'N/A':<10}")

    print("=" * 60)
    print()


def small_tensor_overhead():
    """
    Benchmark: Small Tensor Operations

    Question: Is there overhead for small tensors that makes GPU slower?
    """
    print("=" * 60)
    print("Benchmark 4: Small Tensor Overhead (Matrix Multiplication)")
    print("=" * 60)

    sizes = [2, 4, 8, 16, 32, 64, 128]

    print(f"{'Size':<10} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for size in sizes:
        def matmul_op(device):
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            c = torch.mm(a, b)
            return c

        cpu_mean, _ = benchmark_operation(matmul_op, 'cpu', warmup=2, iterations=100)

        if torch.cuda.is_available():
            gpu_mean, _ = benchmark_operation(matmul_op, 'cuda', warmup=2, iterations=100)
            speedup = cpu_mean / gpu_mean
            slower = "⚠️ GPU SLOWER" if speedup < 1.0 else ""
            print(f"{size:<10} {cpu_mean:<15.6f} {gpu_mean:<15.6f} {speedup:<10.2f}x {slower}")
        else:
            print(f"{size:<10} {cpu_mean:<15.6f} {'N/A':<15} {'N/A':<10}")

    print("\nKey Insight: For very small tensors, GPU kernel launch overhead")
    print("can outweigh computation time. CPU may be faster!")
    print("=" * 60)
    print()


def batched_operations_benchmark():
    """
    Benchmark: Batched Operations

    Question: How does batch size affect GPU utilization?
    """
    print("=" * 60)
    print("Benchmark 5: Batched Matrix Multiplication (Simulating Deep Learning)")
    print("=" * 60)

    batch_sizes = [1, 4, 16, 64, 256, 1024]
    hidden_dim = 512

    print(f"{'Batch Size':<15} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for batch_size in batch_sizes:
        def batched_matmul(device):
            x = torch.randn(batch_size, hidden_dim, device=device)
            w = torch.randn(hidden_dim, hidden_dim, device=device)
            y = torch.mm(x, w)
            return y

        cpu_mean, _ = benchmark_operation(batched_matmul, 'cpu', warmup=3, iterations=20)

        if torch.cuda.is_available():
            gpu_mean, _ = benchmark_operation(batched_matmul, 'cuda', warmup=3, iterations=20)
            speedup = cpu_mean / gpu_mean
            print(f"{batch_size:<15} {cpu_mean:<15.4f} {gpu_mean:<15.4f} {speedup:<10.2f}x")
        else:
            print(f"{batch_size:<15} {cpu_mean:<15.4f} {'N/A':<15} {'N/A':<10}")

    print("\nKey Insight: Larger batch sizes better utilize GPU parallelism!")
    print("=" * 60)
    print()


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("GPU PERFORMANCE BENCHMARKING SUITE")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  WARNING: CUDA not available. GPU benchmarks will be skipped.")

    print("=" * 60)
    print()

    matrix_multiplication_benchmark()
    element_wise_operations_benchmark()
    reduction_operations_benchmark()
    small_tensor_overhead()
    batched_operations_benchmark()

    print("\n" + "=" * 60)
    print("PRACTICE QUESTIONS:")
    print("=" * 60)
    print("""
    1. Run these benchmarks on your A100 GPU. At what matrix size does GPU
       become faster than CPU for matrix multiplication?

    2. Modify the code to benchmark torch.bmm() (batched matrix multiplication)
       vs torch.mm() in a loop. Which is faster for batch processing?

    3. Implement a benchmark for convolution operations (torch.nn.functional.conv2d).
       How does performance scale with image size and batch size?

    4. Add a benchmark that includes CPU->GPU transfer time. How much does
       transfer overhead affect the total time?

    5. Research CUDA streams. Implement a benchmark that shows overlapping
       data transfer with computation using streams.
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()
