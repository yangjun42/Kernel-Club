import time
import torch
import numpy as np
from basic_triton import add_kernel  # Import the kernel directly

# Parameters to adjust
VECTOR_SIZE = 1000  # Size of vectors to add
NUM_RUNS = 10  # Number of times to run the benchmark
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 1024  # Block size for the kernel

def benchmark_triton_kernel():
    # Create input vectors
    a = torch.rand(VECTOR_SIZE, device=DEVICE)
    b = torch.rand(VECTOR_SIZE, device=DEVICE)
    output = torch.empty_like(a)
    n_elements = VECTOR_SIZE
    
    # Calculate grid size
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Warmup
    for _ in range(10):
        add_kernel[(grid,)](
            a,
            b,
            output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(NUM_RUNS):
        add_kernel[(grid,)](
            a,
            b,
            output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    time_per_run = total_time / NUM_RUNS
    
    # Calculate TFLOPS per iteration - each add is 1 FLOP, we do VECTOR_SIZE adds per iteration
    flops_per_iteration = VECTOR_SIZE
    tflops_per_iteration = flops_per_iteration / (time_per_run * 1e12)
    
    print(f"Vector size: {VECTOR_SIZE:,}")
    print(f"Total time: {total_time:.6f} seconds")
    print(f"Time per run: {time_per_run * 1000:.6f} ms")
    print(f"Performance: {tflops_per_iteration:.6f} TFLOPS per iteration")
    
    # Verify correctness
    expected = a + b
    max_diff = torch.max(torch.abs(output - expected)).item()
    print(f"Maximum difference: {max_diff}")
    assert max_diff < 1e-5, f"Results don't match! Max diff: {max_diff}"
    print("Verification successful: Kernel works correctly!")

if __name__ == "__main__":
    print("Running Triton kernel benchmark...")
    benchmark_triton_kernel()