# PyTorch Learning Path - From Basics to Mastery

Welcome to your comprehensive PyTorch learning journey! This curriculum takes you from GPU fundamentals to building production-ready computer vision models on your A100 instance.

## 📚 Learning Path Overview

```
Project 0: Tensor & GPU Basics (Foundation)
    ↓
Project 1: Neural Networks (Building Blocks)
    ↓
Project 2: Training & Optimization (Training Loop)
    ↓
Project 3: Computer Vision (Real Applications)
```

## 🎯 What You'll Master

By completing these 4 progressive projects, you will:

✅ **GPU Programming**: Understand CUDA, memory management, CPU↔GPU transfer overhead
✅ **PyTorch Fundamentals**: Tensors, autograd, nn.Module, datasets, dataloaders
✅ **Network Architecture**: Build custom layers, ResNets, attention mechanisms
✅ **Training Techniques**: Optimizers, learning rate schedules, mixed precision, gradient clipping
✅ **Computer Vision**: CNNs, data augmentation, transfer learning, CIFAR-10 classification
✅ **Optimization**: Profile code, maximize GPU utilization, debug training issues

## 📊 Projects

### Project 0: Tensor & GPU Basics ⚡
**Directory**: `pytorch_00_tensor_gpu_basics/`

**Focus**: GPU fundamentals, CUDA operations, memory profiling

**Key Topics**:
- Creating tensors on CPU/GPU
- Measuring transfer overhead
- Benchmarking matrix multiplication (CPU vs GPU)
- GPU memory management (allocated vs reserved)
- In-place operations for memory efficiency

**Time**: 2-3 hours
**Difficulty**: ⭐ Beginner

**Start Here**:
```bash
cd pytorch_00_tensor_gpu_basics
python src/tensor_basics.py
python src/gpu_benchmarks.py
python src/memory_profiling.py
```

**Expected Outcome**: Understand when GPU acceleration helps and how to manage GPU memory effectively.

---

### Project 1: Neural Networks 🧠
**Directory**: `pytorch_01_neural_networks/`

**Focus**: Building custom modules, understanding layers, initialization, hooks

**Key Topics**:
- Custom nn.Module classes (from scratch)
- Layer types (Linear, Conv, BatchNorm, Dropout, Pooling)
- Weight initialization (Xavier, Kaiming, Orthogonal)
- Hooks for inspecting activations and gradients
- Gradient flow analysis

**Time**: 4-6 hours
**Difficulty**: ⭐⭐ Intermediate

**Start Here**:
```bash
cd pytorch_01_neural_networks
python src/custom_modules.py
python src/layer_playground.py
python src/initialization_study.py
python src/hooks_and_gradients.py
```

**Expected Outcome**: Build any neural network architecture from scratch and debug training issues.

---

### Project 2: Training & Optimization 🚀
**Directory**: `pytorch_02_training_optimization/`

**Focus**: Complete training loops, optimizers, learning rate scheduling, mixed precision

**Key Topics**:
- Training pipeline (train/val split, checkpointing, early stopping)
- Optimizers (SGD, Adam, AdamW) comparison
- Learning rate schedules (cosine, warmup, step decay)
- Automatic Mixed Precision (AMP) for 2x speedup
- Gradient accumulation and clipping
- Distributed training basics (DataParallel, DDP)

**Time**: 4-6 hours
**Difficulty**: ⭐⭐⭐ Advanced

**Start Here**:
```bash
cd pytorch_02_training_optimization
python src/train.py
```

**Expected Outcome**: Train models efficiently with best practices (optimizers, LR schedules, AMP).

---

### Project 3: Computer Vision 📸
**Directory**: `pytorch_03_computer_vision/`

**Focus**: Real-world CV tasks (CNNs, CIFAR-10, transfer learning)

**Key Topics**:
- Build CNNs from scratch
- Train on CIFAR-10 (50k images, 10 classes)
- Data augmentation (RandomCrop, HorizontalFlip, ColorJitter)
- Transfer learning with pretrained ResNet
- Efficient data loading (num_workers, pin_memory)
- Mixed precision training on real datasets

**Time**: 6-8 hours
**Difficulty**: ⭐⭐⭐ Advanced

**Start Here**:
```bash
cd pytorch_03_computer_vision
python src/cnn_from_scratch.py      # Train CNN on CIFAR-10
python src/transfer_learning.py     # Fine-tune pretrained ResNet
```

**Expected Outcome**: Achieve 85-90% accuracy on CIFAR-10 and understand transfer learning.

---

## 🚀 Quick Start

### 1. Set Up Environment

On your Lambda Labs A100 instance:

```bash
# Verify CUDA
nvidia-smi

# Verify PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Install dependencies for a project
cd pytorch_00_tensor_gpu_basics
pip install -r requirements.txt
```

### 2. Follow the Learning Path

**Recommended order**:
1. Start with Project 0 (skip if you're comfortable with PyTorch tensors and GPU operations)
2. Complete Project 1 (essential foundation for modules and layers)
3. Complete Project 2 (training loops and optimization)
4. Complete Project 3 (apply everything to real CV tasks)

### 3. Run Exercises

Each project has multiple Python files with exercises:

```bash
# Example: Project 0
cd pytorch_00_tensor_gpu_basics
python src/tensor_basics.py          # Learn tensor operations
python src/gpu_benchmarks.py         # Benchmark CPU vs GPU
python src/memory_profiling.py       # Profile GPU memory
```

### 4. Practice Questions

Each script ends with practice questions. **Complete them!** They solidify your understanding.

---

## 💡 Learning Tips

### Do's ✅
- **Run every script**: Don't just read, execute the code
- **Modify and experiment**: Change hyperparameters, architectures
- **Complete practice questions**: They're designed to deepen understanding
- **Use your A100**: Take advantage of the powerful GPU
- **Track experiments**: Use MLflow/Weights & Biases for logging
- **Read error messages carefully**: PyTorch errors are very informative

### Don'ts ❌
- Don't skip Project 0 if you're new to PyTorch
- Don't rush through exercises
- Don't ignore practice questions
- Don't forget to check GPU utilization (`nvidia-smi dmon`)
- Don't train without validation set

---

## 📈 Progress Tracking

Track your progress:

- [ ] **Project 0**: Tensor & GPU Basics
  - [ ] Tensor operations and device management
  - [ ] GPU benchmarking (CPU vs GPU)
  - [ ] Memory profiling and optimization
  - [ ] Complete 5 practice questions

- [ ] **Project 1**: Neural Networks
  - [ ] Custom nn.Module classes
  - [ ] Layer experimentation
  - [ ] Weight initialization study
  - [ ] Hooks and gradient inspection
  - [ ] Complete 7 practice questions per script

- [ ] **Project 2**: Training & Optimization
  - [ ] Complete training pipeline
  - [ ] Optimizer comparison
  - [ ] Learning rate scheduling
  - [ ] Mixed precision training
  - [ ] Gradient accumulation

- [ ] **Project 3**: Computer Vision
  - [ ] Train CNN from scratch (>80% on CIFAR-10)
  - [ ] Transfer learning with ResNet (>85% on CIFAR-10)
  - [ ] Data augmentation experiments
  - [ ] DataLoader optimization
  - [ ] Mixed precision training

---

## 🛠️ Common Issues & Solutions

### Issue 1: "CUDA out of memory"
**Solution**:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision (AMP)
- Clear cache: `torch.cuda.empty_cache()`

### Issue 2: "Expected all tensors to be on the same device"
**Solution**:
```python
model = model.to(device)
data = data.to(device)
```

### Issue 3: Slow training
**Solution**:
- Check GPU utilization: `nvidia-smi dmon`
- Increase batch size (better GPU utilization)
- Use more DataLoader workers
- Enable pin_memory=True
- Use mixed precision (AMP)

### Issue 4: Model not learning
**Solution**:
- Check learning rate (too high or too low)
- Verify loss is being computed correctly
- Check gradient flow (use hooks from Project 1)
- Ensure model is in train mode: `model.train()`
- Verify data normalization

---

## 📖 Additional Resources

### Official Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)

### Papers & Books
- **Deep Learning** by Goodfellow, Bengio, Courville
- **Dive into Deep Learning** (d2l.ai) - Interactive PyTorch book
- Key papers: ResNet, Attention is All You Need, EfficientNet

### Community
- [PyTorch Forums](https://discuss.pytorch.org/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)
- [Papers with Code](https://paperswithcode.com/) - SOTA benchmarks

---

## 🎓 What's Next?

After completing these 4 projects, you'll be ready for:

### Advanced Topics
- **Transformers**: Attention mechanisms, BERT, GPT, Vision Transformers
- **GANs**: Generative Adversarial Networks for image generation
- **Object Detection**: YOLO, Faster R-CNN, RetinaNet
- **Segmentation**: U-Net, Mask R-CNN, DeepLab
- **Reinforcement Learning**: DQN, Policy Gradients, PPO

### Practical Projects
- Deploy models with TorchServe or ONNX Runtime
- Build web apps with FastAPI + PyTorch
- Compete on Kaggle
- Contribute to open-source PyTorch projects
- Build your own research project

### Optimization & Production
- Model quantization (INT8) for faster inference
- TorchScript for production deployment
- Multi-GPU training with DDP
- Mixed precision training (FP16, BF16)
- Model profiling and optimization

---

## 🏆 Certification Goals

By the end, you should be able to:

1. ✅ **Implement any architecture** from a paper (ResNet, Transformer, etc.)
2. ✅ **Train efficiently** on GPUs with best practices
3. ✅ **Debug training issues** (vanishing gradients, overfitting, etc.)
4. ✅ **Optimize performance** (data loading, mixed precision, profiling)
5. ✅ **Build real applications** (image classification, object detection)
6. ✅ **Deploy models** to production

---

## 📊 Estimated Timeline

- **Fast track** (already know ML): 2-3 days (8-10 hours/day)
- **Standard pace** (some ML background): 1-2 weeks (2-3 hours/day)
- **Comprehensive** (new to deep learning): 3-4 weeks (2 hours/day)

**Remember**: Quality > Speed. Master each concept before moving on.

---

## 🎯 Final Advice

> "The only way to learn deep learning is by doing. Run the code, break it, fix it, and understand why it works."

**Your A100 GPU is a powerful tool** - use it to experiment freely! Don't be afraid to:
- Try crazy architectures
- Break things and debug
- Run long experiments overnight
- Compare different approaches

**Good luck on your PyTorch journey!** 🚀

---

*Built with ❤️ for learning PyTorch on Lambda Labs A100*
