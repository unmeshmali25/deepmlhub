# PyTorch Computer Vision

Apply everything you've learned to real computer vision tasks: CNNs, CIFAR-10, and transfer learning.

## Learning Objectives

- Build CNNs from scratch for image classification
- Apply data augmentation to improve generalization
- Use transfer learning with pretrained models (ResNet)
- Optimize DataLoader for maximum GPU utilization
- Train with mixed precision for faster training
- Implement practical CV workflows

## Prerequisites

Complete Projects 0-2 for foundational knowledge of:
- GPU operations and memory management
- Neural network modules and layers
- Training loops and optimization

## Project Structure

```
pytorch_03_computer_vision/
├── src/
│   ├── cnn_from_scratch.py      # Train CNN on CIFAR-10
│   ├── transfer_learning.py     # Fine-tune pretrained ResNet
│   ├── data_augmentation.py     # Augmentation techniques
│   └── efficient_dataloading.py # DataLoader optimization
├── data/                        # CIFAR-10 dataset (auto-downloaded)
├── models/                      # Saved checkpoints
├── params.yaml
└── README.md
```

## Exercises

### Exercise 1: CNN from Scratch (`cnn_from_scratch.py`)

**Run:**
```bash
python src/cnn_from_scratch.py
```

**What you'll build:**
- 3-block CNN architecture (Conv → BN → ReLU → Pool)
- Train on CIFAR-10 (50k train, 10k test images)
- Data augmentation (random crop, horizontal flip)
- Achieve ~80-85% test accuracy

**Architecture:**
```
Input (3x32x32)
→ Conv Block 1 (32 filters) → MaxPool
→ Conv Block 2 (64 filters) → MaxPool
→ Conv Block 3 (128 filters) → MaxPool
→ Fully Connected → 10 classes
```

**Key techniques:**
- BatchNorm after each conv layer
- Dropout for regularization
- Data augmentation for better generalization
- Cosine annealing LR schedule

**Practice Questions:**
1. Modify architecture: Add more layers, try different filter sizes
2. Experiment with dropout rates: 0.0, 0.2, 0.5. How does it affect training?
3. Remove BatchNorm. How does training change?
4. Try different augmentations: rotation, color jitter, cutout
5. Calculate total parameters. How does it compare to ResNet?

---

### Exercise 2: Transfer Learning (`transfer_learning.py`)

**Run:**
```bash
python src/transfer_learning.py
```

**What you'll learn:**
- Load pretrained ResNet18 (trained on ImageNet)
- Freeze feature extraction layers
- Replace final FC layer for CIFAR-10 (1000 → 10 classes)
- Fine-tune only the classifier
- Achieve 85-90% accuracy in fewer epochs

**Two-stage fine-tuning:**
1. **Stage 1**: Freeze backbone, train only final layer (fast)
2. **Stage 2**: Unfreeze all layers, fine-tune entire network (better accuracy)

**Practice Questions:**
1. Compare training time: CNN from scratch vs transfer learning
2. Try different backbones: ResNet50, EfficientNet, Vision Transformer
3. Implement discriminative layer-wise learning rates (higher LR for later layers)
4. Experiment with freezing different numbers of layers
5. Use transfer learning on a different dataset (e.g., your own images)

**Expected insights:**
- Transfer learning converges much faster
- Pretrained features generalize well across datasets
- Fine-tuning all layers gives best accuracy (but slower)
- Requires less training data to achieve good performance

---

### Exercise 3: Data Augmentation

**Augmentation techniques:**
1. **Spatial**: RandomCrop, HorizontalFlip, Rotation, Affine
2. **Color**: ColorJitter, Grayscale, Normalization
3. **Advanced**: Cutout, MixUp, CutMix, RandAugment

**Example:**
```python
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

**Practice:**
- Train with/without augmentation. Compare test accuracy.
- Implement Cutout (randomly mask patches)
- Try AutoAugment or RandAugment
- Visualize augmented images

---

### Exercise 4: Efficient DataLoading

**Optimization techniques:**
1. **num_workers**: Parallel data loading (test 0, 2, 4, 8 workers)
2. **pin_memory=True**: Faster CPU → GPU transfer
3. **persistent_workers=True**: Reuse worker processes (PyTorch 1.7+)
4. **prefetch_factor**: Buffer future batches (default 2)

**Benchmark:**
```python
import time

loader = DataLoader(dataset, batch_size=128, num_workers=4,
                    pin_memory=True, persistent_workers=True)

start = time.time()
for batch in loader:
    data, labels = batch
    data = data.cuda(non_blocking=True)
    # Your training code here
print(f"Time: {time.time() - start:.2f}s")
```

**Practice Questions:**
1. Benchmark different num_workers values. What's optimal for A100?
2. Compare with/without pin_memory and persistent_workers
3. Implement custom Dataset that preprocesses on-the-fly
4. Use DALI or ffcv for maximum data loading speed

---

### Exercise 5: Mixed Precision Training

**Using AMP on CIFAR-10:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()

    with autocast():  # Use float16 for forward pass
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits on A100:**
- ~2x faster training
- 40-50% less memory usage
- Same final accuracy (with proper loss scaling)

**Practice:**
- Benchmark training time with/without AMP
- Monitor GPU memory usage
- Test on larger models (ResNet50, EfficientNet)
- Compare float16 vs bfloat16 (if available)

---

## Real-World Challenges

### Challenge 1: Achieve 90%+ Accuracy on CIFAR-10
Use everything you've learned:
- Best architecture (ResNet, WideResNet, EfficientNet)
- Strong augmentation (AutoAugment, Cutout)
- Optimal hyperparameters (LR schedule, weight decay)
- Mixed precision training
- Longer training (200+ epochs)

### Challenge 2: Custom Dataset Classification
Create a custom dataset (e.g., your own photos):
- Organize into folders (one per class)
- Use ImageFolder dataset
- Apply transfer learning
- Achieve high accuracy

### Challenge 3: Real-Time Inference
- Export model to TorchScript or ONNX
- Optimize for inference (remove dropout, fuse BN)
- Benchmark FPS on A100
- Build simple web app with FastAPI

### Challenge 4: Multi-GPU Training
If you have access to multiple GPUs:
- Use DistributedDataParallel (DDP)
- Compare training time vs single GPU
- Implement gradient accumulation for very large batches
- Monitor GPU utilization

---

## Practical Tips

### Debugging Checklist
- ✅ Model on correct device? `model.to(device)`
- ✅ Data on correct device? `data.to(device)`
- ✅ Model in train/eval mode? `model.train()` / `model.eval()`
- ✅ Gradients zeroed? `optimizer.zero_grad()`
- ✅ Learning rate reasonable? (Too high → NaN, too low → slow)
- ✅ Loss decreasing? Plot train/val loss curves
- ✅ Data augmentation only on training set?

### Common Issues

**Issue**: "CUDA out of memory"
**Solutions**:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision (AMP)
- Use gradient checkpointing for very deep models

**Issue**: Training loss decreases but val loss increases
**Solutions**:
- Overfitting! Add regularization (dropout, weight decay)
- Stronger data augmentation
- Early stopping
- Use more training data

**Issue**: Very slow data loading
**Solutions**:
- Increase num_workers (test 4, 8, 16)
- Enable pin_memory=True
- Use SSD instead of HDD for data storage
- Preprocess and cache data

---

## Key Takeaways

✅ **CNNs excel at image tasks**: Spatial hierarchies, translation invariance
✅ **Data augmentation is critical**: Improves generalization significantly
✅ **Transfer learning saves time**: Pretrained features generalize well
✅ **BatchNorm stabilizes training**: Allows higher learning rates
✅ **Optimize DataLoader**: num_workers, pin_memory for max throughput
✅ **AMP provides free speedup**: 2x faster on modern GPUs (A100, V100)
✅ **Monitor both train and val**: Catch overfitting early

## Next Steps

You've completed all 4 practice projects! 🎉

**What's next:**
1. **Implement state-of-the-art models**: Vision Transformers, EfficientNet, ConvNeXt
2. **Explore advanced topics**: Object detection (YOLO, Faster R-CNN), segmentation (U-Net, Mask R-CNN)
3. **Compete on Kaggle**: Apply your skills to real competitions
4. **Build real projects**: Deploy models in production, build web apps
5. **Contribute to open source**: PyTorch, torchvision, timm

## Resources

- [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)
- [Papers with Code](https://paperswithcode.com/) - State-of-the-art benchmarks
- [timm library](https://github.com/huggingface/pytorch-image-models) - 100s of pretrained models
- [Albumentations](https://albumentations.ai/) - Advanced augmentation library
- [MMDetection](https://github.com/open-mmlab/mmdetection) - Object detection framework

---

**Congratulations on mastering PyTorch!** 🚀

You now have the skills to:
- Build any neural network architecture
- Train efficiently on GPUs
- Debug and optimize models
- Apply deep learning to real-world problems

Keep learning, keep building!
