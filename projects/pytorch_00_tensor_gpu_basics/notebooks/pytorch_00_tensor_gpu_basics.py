# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch
import time
import sys

# %% [markdown]
# # Checking CUDA Availability

# %%
print(f"Pytorch version : {torch.__version__}")

# %%
print(f"CUDA available? : {torch.cuda.is_available()}")

# %%
print(f"CUDA version : {torch.version.cuda}")

# %%
print(f"Number of GPUs : {torch.cuda.device_count()}")
print(f"GPU memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# %% [markdown]
# # Tensor Creation Basics

# %%
cpu_tensor = torch.randn(3,3)

# %%
print(f"CPU tensor : \n{cpu_tensor}")
print(f"Device : {cpu_tensor.device}")
print(f"Data Type : {cpu_tensor.dtype}")

# %% [markdown]
# ### Creating tensor directly on GPU

# %%
if torch.cuda.is_available(): 
    gpu_tensor = torch.randn(3,3, device = 'cuda')
    print(f"GPU Tensor : \n{gpu_tensor}")
    print(f"Device : {gpu_tensor.device}")
    print(f"Data Type : {gpu_tensor.dtype}")

# %%
# Alternate way to do that 

# %%
cpu_to_gpu = torch.randn(3,3).cuda()
print(f"Device : {cpu_to_gpu.device}")

# %% [markdown]
# # Measure Device Transfer Overhead

# %% [raw]
# Questions to answer 
# - How expensive is CPU-GPU Transfer? 
# - Does tensor time scale linearly with tensor size?
# - Should you batch multiple small transfers or do them individually? 

# %%
sizes = [100, 1000, 10000, 100000]

# %%
print(f"{'Size':<15} {'CPU->GPU (ms)':<20} {'GPU->CPU (ms)':<20}")
for size in sizes: 
    cpu_tensor = torch.randn(size,size)

    # Measure CPU > GPU transfer
    start = time.time()
    gpu_tensor = cpu_tensor.cuda()
    torch.cuda.synchronize()
    cpu_to_gpu_time = (time.time() - start) * 1000

    # Measure GPU to CPU transfer 
    start = time.time()
    back_to_cpu = gpu_tensor.cpu()
    gpu_to_cpu_time = (time.time() - start) * 1000

    print(f"{size:<15} {cpu_to_gpu_time:<20.4f} {gpu_to_cpu_time:<20.4f}")

# %% [raw]
# Something is wrong with the above numbers. 
# #TODO
# I'll come back to these numbers later and dig deeper into them. 
#
# Let's move on to the next step

# %% [markdown]
# # Tensor Operations on GPU
# ```markdown
# Questions to explore
# 1. Do operations require tensors to be on the same device? 
# 2. What happens if you mix CPU and GPU tensors? 
# 3. How does broadcasting work on GPU? 

# %%
# Create tensors on GPU
device = 'cuda'
a = torch.randn(3,3, device=device)
b = torch.randn(3,3, device=device)

# %%
print("Addition: ")
c = a + b 
print(f"Result device : {c.device}")

# %%
print("Matrix Multiplication: ")
c = torch.mm(a,b)
print(f"Result device : {c.device}")

# %%
# Broadcasting 
print("Broadcasting (tensor + scalar)")
e = a + 5.0
print(f"Result shape: {e.shape}, device: {e.device}")

# %%
# Demonstrate device mismatch error
if torch.cuda.is_available(): 
    print(f"\n Attempting CPU + GPU operation: ")
    try: 
        cpu_tensor = torch.randn(3,3)
        gpu_tensor = torch.randn(3,3, device='cuda')
        result = cpu_tensor + gpu_tensor
    except RuntimeError as e: 
        print(f"Error: {e}")
        print(f"All tensors in an operation must be on the same device")
        
    
