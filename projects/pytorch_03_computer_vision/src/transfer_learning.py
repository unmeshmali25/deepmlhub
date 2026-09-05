"""
Transfer Learning with Pretrained Models
=========================================

Fine-tune a pretrained ResNet on CIFAR-10.

Run: python src/transfer_learning.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


def create_transfer_model(num_classes=10, freeze_features=True):
    """
    Create transfer learning model from pretrained ResNet.

    Args:
        num_classes: Number of output classes
        freeze_features: If True, freeze feature extractor layers
    """
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze feature extraction layers
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def main():
    """Transfer learning on CIFAR-10."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Transforms (ResNet expects 224x224 images)
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    # Load CIFAR-10
    print("Loading CIFAR-10...")
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # Create model
    print("\nLoading pretrained ResNet18...")
    model = create_transfer_model(num_classes=10, freeze_features=True).to(device)

    # Only train the final layer
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining only final layer (feature extractor frozen)...")
    print("=" * 60)

    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {train_loss/(batch_idx+1):.3f}, '
                      f'Acc: {100.*correct/total:.2f}%')

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100. * test_correct / test_total
        print(f"Epoch {epoch}: Test Accuracy = {test_acc:.2f}%\n")

    print("=" * 60)
    print("\nTransfer learning demonstrates:")
    print("✓ Faster convergence (pretrained features)")
    print("✓ Better accuracy with less data")
    print("✓ Common in computer vision tasks")
    print("\nNext: Unfreeze earlier layers and fine-tune entire network!")


if __name__ == "__main__":
    main()
