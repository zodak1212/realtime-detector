"""
train.py
========
Two-phase training for the emotion detection model.

Phase 1: Backbone frozen, train classifier head only
Phase 2: Everything unfrozen, fine-tune with differential learning rates

Usage:
  python train.py --train-dir ./data/train --test-dir ./data/test
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

from model import EmotionNet, EMOTION_LABELS, build_model


def get_train_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_test_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion, device, print_report=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="  Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if print_report:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=EMOTION_LABELS, digits=3))

    return running_loss / total, correct / total, all_preds, all_labels


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load datasets
    train_dataset = datasets.ImageFolder(args.train_dir, transform=get_train_transforms())
    test_dataset = datasets.ImageFolder(args.test_dir, transform=get_test_transforms())

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Test samples:     {len(test_dataset)}")
    print(f"Classes:          {train_dataset.classes}")

    # Class-weighted sampling
    class_counts = np.bincount([label for _, label in train_dataset])
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for _, label in train_dataset]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Build model
    model = build_model(device)

    # Class-weighted loss
    weight_tensor = torch.FloatTensor(class_weights / class_weights.sum() * len(class_weights)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # History
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    os.makedirs(args.output_dir, exist_ok=True)

    # ═══════════════════════════════════════
    # PHASE 1: Head only
    # ═══════════════════════════════════════

    print("\n" + "=" * 50)
    print("PHASE 1: Training classifier head (backbone frozen)")
    print("=" * 50)

    model.freeze_backbone()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.head_lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    phase1_epochs = max(args.epochs // 3, 5)

    for epoch in range(phase1_epochs):
        print(f"\nEpoch {epoch+1}/{phase1_epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'emotion_labels': EMOTION_LABELS,
            }, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  * New best model (val_acc: {val_acc:.4f})")

    # ═══════════════════════════════════════
    # PHASE 2: Full fine-tune
    # ═══════════════════════════════════════

    print("\n" + "=" * 50)
    print("PHASE 2: Fine-tuning all layers")
    print("=" * 50)

    model.unfreeze_backbone()

    optimizer = torch.optim.Adam(
        model.get_optimizer_param_groups(
            head_lr=args.finetune_lr,
            backbone_lr=args.finetune_lr / 10
        ),
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - phase1_epochs, eta_min=1e-6
    )

    patience_counter = 0

    for epoch in range(args.epochs - phase1_epochs):
        print(f"\nEpoch {phase1_epochs + epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': phase1_epochs + epoch,
                'emotion_labels': EMOTION_LABELS,
            }, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  * New best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping — no improvement for {args.patience} epochs.")
                break

    # ═══════════════════════════════════════
    # Final evaluation
    # ═══════════════════════════════════════

    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)

    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, _, all_preds, all_labels = evaluate(model, test_loader, criterion, device, print_report=True)
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    save_plots(history, all_labels, all_preds, args.output_dir)
    print(f"\nOutputs saved to: {os.path.abspath(args.output_dir)}")


def save_plots(history, labels, preds, output_dir):
    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm_norm, cmap='Blues')
    ax.set(xticks=range(len(EMOTION_LABELS)), yticks=range(len(EMOTION_LABELS)),
           xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
           ylabel='True', xlabel='Predicted', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(len(EMOTION_LABELS)):
        for j in range(len(EMOTION_LABELS)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    print("  Saved training_curves.png and confusion_matrix.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--test-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--finetune-lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()
    train(args)
