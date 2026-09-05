# PyTorch Training & Optimization

Master complete training loops, optimizers, learning rate scheduling, and mixed precision training.

## Learning Objectives

- Build complete training pipelines with validation and checkpointing
- Compare optimizers (SGD, Adam, AdamW)
- Implement learning rate schedules (cosine, warmup, step decay)
- Use Automatic Mixed Precision (AMP) for faster training
- Understand gradient accumulation and clipping
- Basics of distributed training (DataParallel, DistributedDataParallel)

## Exercises

### 1. Complete Training Pipeline (`train.py`)
- Full training loop with train/validation split
- Model checkpointing (save best model)
- Early stopping
- MLflow logging integration

**Practice**: Add early stopping, gradient clipping, learning rate warmup

### 2. Optimizer Comparison
Compare SGD, Adam, AdamW, RMSprop on same task. Measure convergence speed and final accuracy.

**Key questions**: When to use each? How does momentum help? What's weight decay?

### 3. Learning Rate Scheduling
- Step decay: reduce LR every N epochs
- Cosine annealing: smooth LR decay
- Warmup: gradually increase LR at start
- ReduceLROnPlateau: reduce when metric plateaus

**Practice**: Plot LR over epochs, compare convergence

### 4. Mixed Precision Training (AMP)
Use `torch.cuda.amp.autocast()` and `GradScaler` for:
- 2x faster training on A100
- 50% memory savings
- Same accuracy (with proper loss scaling)

**Practice**: Benchmark training speed with/without AMP

### 5. Gradient Accumulation
Simulate large batch sizes with small batches:
```python
accumulation_steps = 4
for i, (data, target) in enumerate(loader):
    loss = model(data, target) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 6. Distributed Training Basics
- DataParallel: Simple multi-GPU (single machine)
- DistributedDataParallel: Faster, scales to multiple machines

**Run:** `torchrun --nproc_per_node=2 train.py` (if 2 GPUs available)

## Key Takeaways

✅ Adam is default choice (adaptive LR per parameter)
✅ AdamW fixes weight decay in Adam
✅ LR scheduling improves final accuracy
✅ AMP provides free speedup on modern GPUs (A100, V100)
✅ Gradient accumulation simulates large batches
✅ Always monitor train AND validation metrics

## Next Steps

**Project 3**: `pytorch_03_computer_vision` - Apply these techniques to real CV tasks (CIFAR-10, transfer learning)

## Resources

- [PyTorch Optimization](https://pytorch.org/docs/stable/optim.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
