# train_mobilenet_cifar10.py
#
# This script is an all-in-one solution for defining, training, and analyzing
# a MobileNetV2 model on the CIFAR-10 dataset.

# --- 1. IMPORTS ---
import os
import csv
import math
import time
import argparse
import operator
from functools import reduce

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

#model defination 


class BaseBlock(nn.Module):
    # This is the core "inverted residual" block of MobileNetV2. I hope this is clear
    # It expands channels, performs a lightweight depthwise convolution,
    # and then shrinks them back down.
    def __init__(self, input_channel, output_channel, alpha, t=6, downsample=False):
        super(BaseBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.shortcut = (not downsample) and (input_channel == output_channel)

        # 'alpha' is the width multiplier, making the model thinner or fatter.
        effective_input_channel = int(alpha * input_channel)
        effective_output_channel = int(alpha * output_channel)
        
        # 't' is the expansion factor.
        expansion_channel = t * effective_input_channel
        
        self.layers = nn.Sequential(
            # 1. Point-wise conv for expansion
            nn.Conv2d(effective_input_channel, expansion_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(expansion_channel),
            nn.ReLU6(inplace=True),
            
            # 2. Depth-wise conv for efficient filtering
            nn.Conv2d(expansion_channel, expansion_channel, kernel_size=3, stride=self.stride, padding=1, groups=expansion_channel, bias=False),
            nn.BatchNorm2d(expansion_channel),
            nn.ReLU6(inplace=True),
            
            # 3. Point-wise conv to project back down
            nn.Conv2d(expansion_channel, effective_output_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(effective_output_channel)
        )

    def forward(self, inputs):
        x = self.layers(inputs)
        if self.shortcut:
            x = x + inputs
        return x


class MobileNetV2(nn.Module):
    # The full MobileNetV2 architecture
    def __init__(self, output_size=10, alpha=1.0):
        super(MobileNetV2, self).__init__()
        
        # First layer is a standard convolution. Stride=1 for CIFAR-10's small images.
        self.conv0 = nn.Conv2d(3, int(32 * alpha), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(int(32 * alpha))
        
        # The main sequence of inverted residual blocks.
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, alpha, t=1),
            BaseBlock(16, 24, alpha),
            BaseBlock(24, 24, alpha),
            BaseBlock(24, 32, alpha),
            BaseBlock(32, 32, alpha),
            BaseBlock(32, 32, alpha),
            BaseBlock(32, 64, alpha, downsample=True), # Downsample here
            BaseBlock(64, 64, alpha),
            BaseBlock(64, 64, alpha),
            BaseBlock(64, 64, alpha),
            BaseBlock(64, 96, alpha),
            BaseBlock(96, 96, alpha),
            BaseBlock(96, 96, alpha),
            BaseBlock(96, 160, alpha, downsample=True), # And again here
            BaseBlock(160, 160, alpha),
            BaseBlock(160, 160, alpha),
            BaseBlock(160, 320, alpha)
        )
        
        # Final layers before the classifier.
        last_conv_in = int(320 * alpha)
        last_conv_out = 1280
        
        self.conv1 = nn.Conv2d(last_conv_in, last_conv_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(last_conv_out)
        self.fc = nn.Linear(last_conv_out, output_size)
        
        self.weights_init()

    def weights_init(self):
        # Use Kaiming He initialization for stability.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace=True)
        x = self.bottlenecks(x)
        x = F.relu6(self.bn1(self.conv1(x)), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1) # Global Average Pooling
        x = x.view(x.shape[0], -1)      # Flatten
        x = self.fc(x)
        return x

# these are the utilities that are ofcourse needed in the model

def test(model, testloader, criterion, device):
    """Helper function to evaluate the model on the test set."""
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = test_loss / len(testloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# this is the where we start the training

def main(args):
    """Main function to orchestrate the entire training process."""
    start_time = time.time()
    max_val_acc = 0.0

    # Setup device (GPU or CPU)
    use_cuda = torch.cuda.is_available() and args.gpus
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Prepare CIFAR-10 Data
    print("Preparing CIFAR-10 data")
    cifar10_mean, cifar10_std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    trainset = CIFAR10(root='/home/rupesh/test/final/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = CIFAR10(root='/home/rupesh/test/final/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Build Model, Loss, and Optimizer
    print("==> Building MobileNetV2 model...")
    model = MobileNetV2(output_size=10, alpha=args.alpha).to(device)

    if use_cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)

    # Setup Logging
    log_file = f'log_alpha_{args.alpha}.csv'
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

    # The Training Loop
    print(" Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()

        # Log and evaluate at the end of each epoch
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(trainloader)
        val_loss, val_acc = test(model, testloader, criterion, device)
        
        print(f'Epoch: {epoch:03d} | Train Loss: {avg_train_loss:.3f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, val_loss, train_acc, val_acc])
        
        # Save the best model
        if val_acc > max_val_acc:
            print('  -> New best validation accuracy! Saving model...')
            torch.save(model.state_dict(), f'weights_alpha_{args.alpha}_best.pkl')
            max_val_acc = val_acc
            
    print("\n YAYYYY Training complete! Check the output now and try to compress it ")
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Best validation accuracy was: {max_val_acc:.2f}%")

# main function



if __name__ == "__main__":
    # Command-line arguments make the script flexible.
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 MobileNetV2 Training')
    parser.add_argument('--alpha', default=1.0, type=float, help='Width multiplier alpha (e.g., 0.5, 0.75, 1.0)')
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs to train')
    parser.add_argument('--batch-size', default=128, type=int, help='Training batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
    parser.add_argument('--gpus', action='store_true', help='Use GPUs for training if available')
    args = parser.parse_args()
    
    main(args)
