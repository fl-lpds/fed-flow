#!/usr/bin/env python3
"""
Single Node Model Training Script

This script demonstrates how to train a model on a single node, extracted from 
the federated learning codebase. You can easily modify the model and dataset 
configurations to experiment with different setups.

Usage:
    python train_model.py
"""

import sys
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

# Import project utilities
from app.config import config
from app.util import model_utils, data_utils
from app.model.utils import get_available_torch_device


def main():
    """Main training function."""
    
    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    
    # Configuration - Modify these to change model and dataset
    config.model_name = 'VGG'  # Options: 'VGG', 'VGG16'
    config.dataset_name = 'cifar100'  # Options: 'cifar10', 'cifar100'
    config.dataset_path = 'app/dataset/data/'
    
    # Training hyperparameters
    LEARNING_RATE = 0.01
    BATCH_SIZE = 100
    NUM_EPOCHS = 10
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    LR_STEP_SIZE = 20
    LR_GAMMA = 0.1
    
    # Device configuration
    # Automatically detects GPU if available, otherwise uses CPU
    device = get_available_torch_device()
    
    # GPU-specific settings
    USE_GPU = torch.cuda.is_available()
    PIN_MEMORY = USE_GPU  # Pin memory for faster GPU transfer
    
    # Display device information
    print("=" * 50)
    print("Device Configuration:")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  Using Device: {device}")
    print(f"  Pin Memory: {PIN_MEMORY}")
    print("=" * 50)
    
    # Load training dataset
    print("\nLoading datasets...")
    trainset = data_utils.get_trainset()
    trainloader = DataLoader(
        trainset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        pin_memory=PIN_MEMORY  # Faster GPU transfer when using GPU
    )
    
    # Load test dataset
    testset = data_utils.get_testset()
    testloader = DataLoader(
        testset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=PIN_MEMORY  # Faster GPU transfer when using GPU
    )
    
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    print(f"Number of classes: {len(trainset.classes) if hasattr(trainset, 'classes') else 'Unknown'}")
    
    # Initialize Model
    print("\nInitializing model...")
    # Create model - 'Unit' means full model (not split)
    # None means no layer splitting
    # False means not edge-based
    net = model_utils.get_model('Unit', None, device, False)
    net = net.to(device)
    
    # Print model architecture
    print("Model Architecture:")
    print(net)
    print(f"\nTotal parameters: {sum(p.numel() for p in net.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")
    
    # Setup Optimizer, Loss Function, and Scheduler
    print("\nSetting up optimizer, loss function, and scheduler...")
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (SGD with momentum)
    optimizer = optim.SGD(
        net.parameters(), 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=LR_STEP_SIZE, 
        gamma=LR_GAMMA
    )
    
    print(f"Optimizer: SGD(lr={LEARNING_RATE}, momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY})")
    print(f"Scheduler: StepLR(step_size={LR_STEP_SIZE}, gamma={LR_GAMMA})")
    
    # Training loop
    print("\nStarting training...")
    print("=" * 50)
    net.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%, LR={scheduler.get_last_lr()[0]:.6f}')
        
        # Test accuracy after each epoch
        net.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f'Test Accuracy: {test_acc:.2f}%\n')
        
        net.train()
    
    # Final Evaluation
    print("=" * 50)
    print("\nFinal evaluation...")
    final_accuracy = model_utils.test(net, testloader, device, criterion)
    print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")
    
    # Save Model
    print("\nSaving model...")
    model_path = f'./{config.model_name}_{config.dataset_name}_trained.pth'
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': final_accuracy,
        'config': {
            'model_name': config.model_name,
            'dataset_name': config.dataset_name,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
        }
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()

