"""
train_face.py
=============
Train the face emotion model using EfficientNet-B0 on RAF-DB (Folders 1-7).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import EmotionNet, EMOTION_LABELS

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        _, pred = out.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return loss_sum / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            loss_sum += loss.item() * images.size(0)
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            preds_all.extend(pred.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    return loss_sum / total, correct / total, preds_all, labels_all

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths assuming folders are at face_model/train/1, face_model/train/2, etc.
    train_dir = os.path.join(script_dir, "train")
    test_dir = os.path.join(script_dir, "test")

    train_set = datasets.ImageFolder(train_dir, transform=get_train_transforms())
    test_set = datasets.ImageFolder(test_dir, transform=get_test_transforms())

    # Calculate weights for imbalanced RAF-DB folders
    targets = train_set.targets
    class_counts = np.bincount(targets)
    weights = 1.0 / class_counts
    sample_weights = [weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = EmotionNet().to(device)
    criterion = nn.CrossEntropyLoss()
    model_path = os.path.join(script_dir, "model.pth")
    best_acc = 0.0

    print("Phase 1: Training Head Only...")
    model.freeze_backbone()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.head_lr)
    
    for ep in range(5):
        train_one_epoch(model, train_loader, criterion, opt, device)
        vl, va, _, _ = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {ep+1}/5 - Val Acc: {va:.4f}")

    print("\nPhase 2: Fine-Tuning Entire Model...")
    model.unfreeze_backbone()
    opt = torch.optim.Adam(model.get_optimizer_param_groups(args.ft_lr, args.ft_lr / 10))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)

    for ep in range(args.epochs):
        train_one_epoch(model, train_loader, criterion, opt, device)
        vl, va, preds, labels = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {ep+1}/{args.epochs} - Val Acc: {va:.4f}")
        
        scheduler.step(va)
        
        if va > best_acc:
            best_acc = va
            torch.save({'model_state_dict': model.state_dict(), 'val_acc': va}, model_path)
            print(f"  -> Saved new best model ({va:.4f})")

            # Save Classification Report
            report = classification_report(labels, preds, target_names=EMOTION_LABELS, digits=3)
            with open(os.path.join(script_dir, 'best_report.txt'), 'w') as f:
                f.write(report)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--head-lr', type=float, default=1e-3)
    p.add_argument('--ft-lr', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=4) # Set to 0 if running on Windows throws memory errors
    main(p.parse_args())