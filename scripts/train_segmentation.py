"""
Segmentation Training Script - Google Colab Version
Converted from train_mask.ipynb
Trains a segmentation head on top of DINOv2 backbone
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import torchvision
from tqdm import tqdm
import argparse

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = len(value_map)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Get all image files
        self.data_ids = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        print(f"Found {len(self.data_ids)} images in {data_dir}")

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255

        return image, mask


# ============================================================================
# Model: Segmentation Head (ConvNeXt-style)
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )

        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class)


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Dice coefficient (F1 Score) per class and return mean Dice Score."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)

        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10, show_progress=True):
    """Evaluate all metrics on a dataset."""
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []

    model.eval()
    loader = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") if show_progress else data_loader
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            output = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(output.to(device))
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels = labels.squeeze(dim=1).long()

            iou = compute_iou(outputs, labels, num_classes=num_classes)
            dice = compute_dice(outputs, labels, num_classes=num_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)

    model.train()
    return np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies)


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves to '{output_dir}/training_curves.png'")

    # Plot 2: IoU curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_iou'], label='Train IoU', marker='o')
    plt.title('Train IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU', marker='o', color='orange')
    plt.title('Validation IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved IoU curves to '{output_dir}/iou_curves.png'")

    # Plot 3: Dice curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_dice'], label='Train Dice', marker='s')
    plt.title('Train Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice', marker='s', color='green')
    plt.title('Validation Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Dice curves to '{output_dir}/dice_curves.png'")

    # Plot 4: Combined metrics plot
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['train_iou'], label='train', marker='o')
    plt.plot(history['val_iou'], label='val', marker='o')
    plt.title('IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['train_dice'], label='train', marker='s')
    plt.plot(history['val_dice'], label='val', marker='s')
    plt.title('Dice Score vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['train_pixel_acc'], label='train', marker='^')
    plt.plot(history['val_pixel_acc'], label='val', marker='^')
    plt.title('Pixel Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved combined metrics to '{output_dir}/all_metrics_curves.png'")


def save_history_to_file(history, output_dir):
    """Save training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 20 + "TRAINING RESULTS - PHASE 1 BASELINE\n")
        f.write("=" * 80 + "\n\n")

        f.write("Final Metrics (Last Epoch):\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Best Results Across All Epochs:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Per-Epoch History:\n")
        f.write("-" * 105 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 105 + "\n")

        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_iou'][i],
                history['val_iou'][i],
                history['train_dice'][i],
                history['val_dice'][i],
                history['train_pixel_acc'][i],
                history['val_pixel_acc'][i]
            ))
        f.write("=" * 80 + "\n")

    print(f"✓ Saved evaluation metrics to '{filepath}'")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/SWOC26_Training/data/Train',
                        help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default='/content/drive/MyDrive/SWOC26_Training/data/Val',
                        help='Path to validation data directory')
    parser.add_argument('--save_dir', type=str, default='/content/drive/MyDrive/SWOC26_Training/models/baseline',
                        help='Directory to save models and results')
    
    args = parser.parse_args()

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(" " * 25 + "PHASE 1: BASELINE TRAINING")
    print("=" * 80)
    print(f"\n🔧 Configuration:")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Hyperparameters
    batch_size = args.batch_size
    w = int(((960 / 2) // 14) * 14)  # 476 → adjusted for DINOv2 patch size
    h = int(((540 / 2) // 14) * 14)  # 266 → adjusted for DINOv2 patch size
    lr = args.lr
    n_epochs = args.epochs

    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Image size: {h}x{w}")

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    train_stats_dir = os.path.join(args.save_dir, 'train_stats')
    os.makedirs(train_stats_dir, exist_ok=True)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])

    # Create datasets
    print(f"\n📁 Loading datasets...")
    print(f"  Train dir: {args.data_dir}")
    print(f"  Val dir: {args.val_dir}")
    
    trainset = MaskDataset(data_dir=args.data_dir, transform=transform, mask_transform=mask_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    valset = MaskDataset(data_dir=args.val_dir, transform=transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Training samples: {len(trainset)}")
    print(f"  Validation samples: {len(valset)}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    # Load DINOv2 backbone
    print(f"\n🤖 Loading DINOv2 backbone...")
    BACKBONE_SIZE = "small"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)
    print(f"  ✓ Loaded {backbone_name}")

    # Get embedding dimension from backbone
    print(f"\n🔍 Probing backbone architecture...")
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"  Embedding dimension: {n_embedding}")
    print(f"  Patch tokens shape: {output.shape}")
    print(f"  Tokens per image: {output.shape[1]} ({w // 14} x {h // 14})")

    # Create segmentation head
    print(f"\n🏗️  Building segmentation head...")
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    )
    classifier = classifier.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    loss_fct = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    
    print(f"  Loss function: CrossEntropyLoss")
    print(f"  Optimizer: SGD (momentum=0.9)")

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'train_dice': [],
        'val_dice': [],
        'train_pixel_acc': [],
        'val_pixel_acc': []
    }

    # Training loop
    print("\n" + "=" * 80)
    print(" " * 30 + "STARTING TRAINING")
    print("=" * 80 + "\n")

    best_val_iou = 0.0
    
    for epoch in range(n_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"{'='*80}")
        
        # Training phase
        classifier.train()
        train_losses = []

        train_pbar = tqdm(train_loader, desc=f"[Train]", unit="batch", ncols=100)
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

            logits = classifier(output.to(device))
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels = labels.squeeze(dim=1).long()

            loss = loss_fct(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation phase
        classifier.eval()
        val_losses = []

        val_pbar = tqdm(val_loader, desc=f"[Val]  ", unit="batch", ncols=100)
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs, labels = imgs.to(device), labels.to(device)

                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(output.to(device))
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                labels = labels.squeeze(dim=1).long()
                loss = loss_fct(outputs, labels)
                val_losses.append(loss.item())
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Calculate metrics
        print(f"\n  Computing metrics...")
        train_iou, train_dice, train_pixel_acc = evaluate_metrics(
            classifier, backbone_model, train_loader, device, num_classes=n_classes, show_progress=False
        )
        val_iou, val_dice, val_pixel_acc = evaluate_metrics(
            classifier, backbone_model, val_loader, device, num_classes=n_classes, show_progress=False
        )

        # Store history
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_pixel_acc)
        history['val_pixel_acc'].append(val_pixel_acc)

        # Print epoch summary
        print(f"\n  📊 Epoch {epoch+1} Results:")
        print(f"  {'─'*76}")
        print(f"  {'Metric':<20} {'Train':>12} {'Val':>12} {'Δ':>12}")
        print(f"  {'─'*76}")
        print(f"  {'Loss':<20} {epoch_train_loss:>12.4f} {epoch_val_loss:>12.4f} {epoch_val_loss-epoch_train_loss:>+12.4f}")
        print(f"  {'IoU':<20} {train_iou:>12.4f} {val_iou:>12.4f} {val_iou-train_iou:>+12.4f}")
        print(f"  {'Dice Score':<20} {train_dice:>12.4f} {val_dice:>12.4f} {val_dice-train_dice:>+12.4f}")
        print(f"  {'Pixel Accuracy':<20} {train_pixel_acc:>12.4f} {val_pixel_acc:>12.4f} {val_pixel_acc-train_pixel_acc:>+12.4f}")
        print(f"  {'─'*76}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            model_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(classifier.state_dict(), model_path)
            print(f"  💾 New best model saved! (Val IoU: {val_iou:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(classifier.state_dict(), checkpoint_path)
            print(f"  💾 Checkpoint saved: epoch_{epoch+1}.pth")

    # Save final plots and metrics
    print("\n" + "=" * 80)
    print(" " * 25 + "TRAINING COMPLETE!")
    print("=" * 80)
    
    print(f"\n📊 Generating plots and reports...")
    save_training_plots(history, train_stats_dir)
    save_history_to_file(history, train_stats_dir)

    # Save final model
    final_model_path = os.path.join(args.save_dir, "segmentation_head_final.pth")
    torch.save(classifier.state_dict(), final_model_path)
    print(f"✓ Saved final model to '{final_model_path}'")

    # Final summary
    print(f"\n" + "=" * 80)
    print(" " * 30 + "FINAL SUMMARY")
    print("=" * 80)
    print(f"\n  📈 Best Results:")
    print(f"  {'─'*76}")
    print(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})")
    print(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})")
    print(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})")
    print(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})")
    print(f"  {'─'*76}")
    
    print(f"\n  💾 All files saved to: {args.save_dir}")
    print(f"  📊 Training plots: {train_stats_dir}")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
