import torch

def matrix_multiplication_gpu():
    """
    Basic example to multiply two matrices on GPU with PyTorch
    """
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device used: {device}")
    
    # Matrix dimensions
    m, n, k = 1000, 1000, 1000
    
    # Create two random matrices on CPU
    matrix_a = torch.randn(m, n)
    matrix_b = torch.randn(n, k)
    
    print(f"Matrix A dimensions: {matrix_a.shape}")
    print(f"Matrix B dimensions: {matrix_b.shape}")
    
    # Move matrices to GPU
    matrix_a_gpu = matrix_a.to(device)
    matrix_b_gpu = matrix_b.to(device)
    
    # Perform matrix multiplication on GPU
    result_gpu = torch.matmul(matrix_a_gpu, matrix_b_gpu)
    
    # Alternative: you can also use the @ operator
    # result_gpu = matrix_a_gpu @ matrix_b_gpu
    
    print(f"Result dimensions: {result_gpu.shape}")
    print(f"Result on GPU: {result_gpu.device}")
    
    # Move result back to CPU if needed
    result_cpu = result_gpu.cpu()
    
    return result_cpu


def benchmark_gpu_vs_cpu():
    """
    Compare performance between GPU and CPU
    """
    import time
    
    # Matrix dimensions
    size = 2000
    matrix_a = torch.randn(size, size)
    matrix_b = torch.randn(size, size)
    
    # CPU Benchmark
    start_time = time.time()
    result_cpu = torch.matmul(matrix_a, matrix_b)
    cpu_time = time.time() - start_time
    print(f"CPU Time: {cpu_time:.4f} seconds")
    
    # GPU Benchmark (if available)
    if torch.cuda.is_available():
        matrix_a_gpu = matrix_a.cuda()
        matrix_b_gpu = matrix_b.cuda()
        
        # Warm-up
        _ = torch.matmul(matrix_a_gpu, matrix_b_gpu)
        torch.cuda.synchronize()
        
        start_time = time.time()
        result_gpu = torch.matmul(matrix_a_gpu, matrix_b_gpu)
        torch.cuda.synchronize()  # Wait for GPU operation to complete
        gpu_time = time.time() - start_time
        
        print(f"GPU Time: {gpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("CUDA not available for GPU benchmark")


if __name__ == "__main__":
    print("=== GPU Matrix Multiplication ===\n")
    
    # Run basic multiplication
    result = matrix_multiplication_gpu()
    
    print("\n=== GPU vs CPU Benchmark ===\n")
    
    # Compare performance
    benchmark_gpu_vs_cpu()
