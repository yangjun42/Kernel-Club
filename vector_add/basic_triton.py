import triton
import triton.language as tl
import torch


@triton.jit
def add_kernel(
    x_ptr,                      # Pointer to first input tensor (type determined by tensor)
    y_ptr,                      # Pointer to second input tensor (type determined by tensor)
    output_ptr,                 # Pointer to output tensor (type determined by tensor)
    n_elements,                 # Number of elements in the tensors (scalar int)
    BLOCK_SIZE: tl.constexpr,   # Block size for parallelization (compile-time constant)
):
    # Compute the program ID (unique for each kernel instance)
    pid = tl.program_id(axis=0) # Assume 1D grid

    # Compute the start index for this program/block
    # Each program handles BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE

    # Create a range of offsets relative to the block start
    # Shape: [BLOCK_SIZE]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to handle elements potentially beyond the tensor end
    # Shape: [BLOCK_SIZE], type: bool
    mask = offsets < n_elements


    # Load x and y directly using base pointers and offsets.
    # tl.load implicitly handles pointer arithmetic (base_ptr + offsets)
    # and loads data of the correct type associated with the pointer.
    # The mask prevents out-of-bounds memory access.
    x = tl.load(x_ptr + offsets, mask=mask) # Shape: [BLOCK_SIZE], type: tensor dtype
    y = tl.load(y_ptr + offsets, mask=mask) # Shape: [BLOCK_SIZE], type: tensor dtype

    # Perform the element-wise addition
    output = x + y # Shape: [BLOCK_SIZE], type: tensor dtype

    # Store the result back to memory using the base output pointer and offsets.
    # The mask prevents out-of-bounds memory writes.
    tl.store(output_ptr + offsets, output, mask=mask)



def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Add two tensors element-wise using Triton.

    Args:
        x: First input tensor
        y: Second input tensor

    Returns:
        Result of x + y
    """
    # Check that the inputs are compatible
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.is_cuda and y.is_cuda, "Input tensors must be on GPU"
    assert x.dtype == y.dtype, "Input tensors must have the same data type"

    # Output tensor
    output = torch.empty_like(x)

    # Number of elements
    n_elements = output.numel()

    # Block size - adjust as needed for performance, ensure it's reasonable
    # Powers of 2 are common. 1024 is often a good starting point.
    BLOCK_SIZE = 1024

    # Grid size (number of kernel programs to launch)
    # Use ceiling division to ensure all elements are covered
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    # The grid is defined by a tuple, here a 1D grid of size `grid`.
    # Kernel arguments are passed positionally after the grid specification.
    # BLOCK_SIZE is passed as a regular argument because it's used in calculations
    # and also marked as tl.constexpr for compile-time optimization.
    add_kernel[(grid,)](  # Define the grid shape (1D)
        x,                # Pass the tensor directly (Triton gets the data_ptr implicitly)
        y,                # Pass the tensor directly
        output,           # Pass the tensor directly
        n_elements,       # Pass the number of elements
        BLOCK_SIZE=BLOCK_SIZE # Pass BLOCK_SIZE as a keyword or positional argument
                              # Matching the kernel definition
    )
    # Alternatively, passing data_ptr() also works, Triton handles both:
    # add_kernel[(grid,)](
    #     x.data_ptr(),
    #     y.data_ptr(),
    #     output.data_ptr(),
    #     n_elements,
    #     BLOCK_SIZE=BLOCK_SIZE,
    # )

    return output


if __name__ == "__main__":
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA not available, exiting.")
        exit()

    # Create sample input tensors
    try:
        # Larger tensor to see potential performance benefits
        size = 4096
        x = torch.randn(size, size, device=device, dtype=torch.float32)
        y = torch.randn(size, size, device=device, dtype=torch.float32)

        print(f"Input tensor size: {x.shape}")
        print(f"Input tensor dtype: {x.dtype}")

        # Call our triton add function
        output_triton = add(x, y)

        # Verify with PyTorch's native add
        output_torch = x + y

        # Check if results are close
        max_diff = torch.max(torch.abs(output_triton - output_torch)).item()
        print(f"Maximum difference between Triton and PyTorch: {max_diff}")

        # Test if the difference is within an acceptable threshold for float32
        assert max_diff < 1e-5, f"Results don't match! Max diff: {max_diff}"
        print("Verification successful: Triton add kernel works correctly!")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
