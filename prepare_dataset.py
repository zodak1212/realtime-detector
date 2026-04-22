"""
prepare_dataset.py
==================
Prepares facial emotion datasets for training.

Supports:
  1. FER+ corrected labels (RECOMMENDED)
     - Download fer2013.csv from Kaggle
     - Download FER+ labels from https://github.com/microsoft/FERPlus
     - Run: python prepare_dataset.py --ferplus --csv fer2013.csv --ferplus-labels fer2013new.csv

  2. FER2013 original CSV
     - Run: python prepare_dataset.py --csv fer2013.csv

  3. Existing folder-based dataset
     - Run: python prepare_dataset.py --folder path/to/train --test-folder path/to/test
"""

import os
import argparse
import numpy as np
import csv
from PIL import Image
from tqdm import tqdm
from collections import Counter


EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# FER+ label columns: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, NF
# We map these to our standard 7 labels (dropping contempt, unknown, NF)
FERPLUS_TO_STANDARD = {
    0: 6,  # neutral -> neutral
    1: 3,  # happiness -> happy
    2: 5,  # surprise -> surprise
    3: 4,  # sadness -> sad
    4: 0,  # anger -> angry
    5: 1,  # disgust -> disgust
    6: 2,  # fear -> fear
}


def prepare_ferplus(csv_path: str, ferplus_labels_path: str, output_dir: str):
    """
    Parse FER2013 CSV with FER+ corrected labels.

    FER+ provides 10 annotator votes per image. We use majority vote
    to determine the label, and discard images where there's no clear
    consensus (ties or too many 'unknown'/'NF' votes).
    """
    print(f"Reading FER2013 images from {csv_path}...")
    print(f"Reading FER+ labels from {ferplus_labels_path}...")

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    for split_dir in [train_dir, test_dir]:
        for label in EMOTION_LABELS:
            os.makedirs(os.path.join(split_dir, label), exist_ok=True)

    # Read FER2013 pixels
    with open(csv_path, 'r') as f:
        fer_rows = list(csv.DictReader(f))

    # Read FER+ labels
    with open(ferplus_labels_path, 'r') as f:
        ferplus_rows = list(csv.reader(f))
        # Skip header if present
        if ferplus_rows[0][0].lower() in ['usage', 'emotion']:
            ferplus_rows = ferplus_rows[1:]

    if len(fer_rows) != len(ferplus_rows):
        print(f"WARNING: FER2013 has {len(fer_rows)} rows but FER+ has {len(ferplus_rows)} rows.")
        print("Using the minimum of both.")

    counters = {"train": 0, "test": 0}
    skipped = 0

    for i in tqdm(range(min(len(fer_rows), len(ferplus_rows))), desc="Processing images"):
        fer_row = fer_rows[i]
        ferplus_row = ferplus_rows[i]

        # FER+ has 10 votes across categories:
        # neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, NF
        try:
            votes = [int(v) for v in ferplus_row[2:12]]  # columns 2-11 are the votes
        except (ValueError, IndexError):
            # Try alternative parsing — some FER+ formats differ
            try:
                votes = [int(v) for v in ferplus_row[-10:]]
            except (ValueError, IndexError):
                skipped += 1
                continue

        # Only consider the 7 standard emotions (indices 0-6, skip contempt/unknown/NF)
        standard_votes = {FERPLUS_TO_STANDARD[j]: votes[j] for j in range(7)}

        # Majority vote
        best_emotion = max(standard_votes, key=standard_votes.get)
        best_votes = standard_votes[best_emotion]
        total_valid_votes = sum(standard_votes.values())

        # Skip if no clear consensus (less than 30% agreement)
        if total_valid_votes == 0 or best_votes / total_valid_votes < 0.3:
            skipped += 1
            continue

        # Convert pixels to image
        pixels = fer_row['pixels']
        pixel_values = np.array([int(p) for p in pixels.split()], dtype=np.uint8)
        img = pixel_values.reshape(48, 48)
        img = Image.fromarray(img, mode='L')

        # Determine split
        usage = fer_row.get('Usage', ferplus_row[0] if ferplus_row[0] in ['Training', 'PublicTest', 'PrivateTest'] else 'Training')
        if usage == 'Training':
            split = "train"
            split_dir = train_dir
        else:
            split = "test"
            split_dir = test_dir

        label = EMOTION_LABELS[best_emotion]
        filename = f"{label}_{counters[split]:05d}.png"
        img.save(os.path.join(split_dir, label, filename))
        counters[split] += 1

    print(f"\nDone! Saved {counters['train']} training and {counters['test']} test images.")
    print(f"Skipped {skipped} ambiguous images (no clear consensus).")
    print(f"Output: {output_dir}")
    print_distribution(train_dir, "Training")
    print_distribution(test_dir, "Test")


def prepare_from_csv(csv_path: str, output_dir: str):
    """
    Parse standard FER2013 CSV and save images into train/test folder structure.
    """
    print(f"Reading {csv_path}...")

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    for split_dir in [train_dir, test_dir]:
        for label in EMOTION_LABELS:
            os.makedirs(os.path.join(split_dir, label), exist_ok=True)

    counters = {"train": 0, "test": 0}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in tqdm(rows, desc="Processing images"):
        emotion_idx = int(row['emotion'])
        pixels = row['pixels']
        usage = row['Usage']

        pixel_values = np.array([int(p) for p in pixels.split()], dtype=np.uint8)
        img = pixel_values.reshape(48, 48)
        img = Image.fromarray(img, mode='L')

        if usage == 'Training':
            split = "train"
            split_dir = train_dir
        else:
            split = "test"
            split_dir = test_dir

        label = EMOTION_LABELS[emotion_idx]
        filename = f"{label}_{counters[split]:05d}.png"
        img.save(os.path.join(split_dir, label, filename))
        counters[split] += 1

    print(f"\nDone! Saved {counters['train']} training and {counters['test']} test images.")
    print(f"Output: {output_dir}")
    print_distribution(train_dir, "Training")
    print_distribution(test_dir, "Test")


def prepare_from_folders(train_folder: str, test_folder: str):
    """Verify existing folder-based dataset."""
    print(f"Verifying folder dataset...")
    print_distribution(train_folder, "Training")
    if test_folder:
        print_distribution(test_folder, "Test")
    print(f"\nYour dataset is ready to use:")
    print(f"  Train: {os.path.abspath(train_folder)}")
    if test_folder:
        print(f"  Test:  {os.path.abspath(test_folder)}")


def print_distribution(data_dir: str, split_name: str):
    """Print class distribution for a dataset split."""
    print(f"\n{split_name} distribution:")
    total = 0
    for label in EMOTION_LABELS:
        label_dir = os.path.join(data_dir, label)
        if os.path.exists(label_dir):
            count = len([f for f in os.listdir(label_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            print(f"  {label:>10s}: {count:5d}")
            total += count
        else:
            print(f"  {label:>10s}: (folder not found)")
    print(f"  {'TOTAL':>10s}: {total:5d}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare emotion dataset")
    parser.add_argument('--csv', type=str, help="Path to fer2013.csv")
    parser.add_argument('--ferplus', action='store_true', help="Use FER+ corrected labels")
    parser.add_argument('--ferplus-labels', type=str, help="Path to FER+ label file (fer2013new.csv)")
    parser.add_argument('--folder', type=str, help="Path to training image folder")
    parser.add_argument('--test-folder', type=str, help="Path to test image folder")
    parser.add_argument('--output', type=str, default='./data', help="Output directory")

    args = parser.parse_args()

    if args.ferplus:
        if not args.csv or not args.ferplus_labels:
            print("FER+ mode requires both --csv and --ferplus-labels")
            print("  python prepare_dataset.py --ferplus --csv fer2013.csv --ferplus-labels fer2013new.csv")
            exit(1)
        prepare_ferplus(args.csv, args.ferplus_labels, args.output)
    elif args.csv:
        prepare_from_csv(args.csv, args.output)
    elif args.folder:
        prepare_from_folders(args.folder, args.test_folder)
    else:
        print("Usage:")
        print("  FER+ (recommended):")
        print("    python prepare_dataset.py --ferplus --csv fer2013.csv --ferplus-labels fer2013new.csv")
        print("")
        print("  FER2013 original:")
        print("    python prepare_dataset.py --csv fer2013.csv")
        print("")
        print("  Existing folders:")
        print("    python prepare_dataset.py --folder path/to/train --test-folder path/to/test")
