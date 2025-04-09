import time
import torch
import numpy as np
from basic_triton import triton_add

# Parameters to adjust
VECTOR_SIZE = 1000  # Size of vectors to add
NUM_RUNS = 10  # Number of times to run the benchmark
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def benchmark_triton_add():
    # Create input vectors
    a = torch.rand(VECTOR_SIZE, device=DEVICE)
    b = torch.rand(VECTOR_SIZE, device=DEVICE)
    
    # Warmup
    for _ in range(10):
        c = triton_add(a, b)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(NUM_RUNS):
        c = triton_add(a, b)
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

if __name__ == "__main__":
    print("Running Triton vector add benchmark...")
    benchmark_triton_add()