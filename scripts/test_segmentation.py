"""
Segmentation Validation/Test Script - Google Colab Version
Converted from val_mask.ipynb
Evaluates a trained segmentation head on validation/test data and saves predictions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm
import time

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
    img = np.clip(img, 0, 255).astype(np.uint8)
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

# Class names for visualization
class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

n_classes = len(value_map)

# Color palette for visualization (10 distinct colors)
color_palette = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    """Convert a class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


# ============================================================================
# Dataset (with support for test images without masks)
# ============================================================================

class MaskDataset(Dataset):
    """Dataset that handles both validation (with masks) and test (without masks)"""
    def __init__(self, data_dir, transform=None, mask_transform=None, has_masks=True):
        self.has_masks = has_masks
        
        if has_masks:
            # Validation mode: data_dir/Color_Images and data_dir/Segmentation
            self.image_dir = os.path.join(data_dir, 'Color_Images')
            self.masks_dir = os.path.join(data_dir, 'Segmentation')
        else:
            # Test mode: images directly in data_dir
            self.image_dir = data_dir
            self.masks_dir = None
        
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Get all image files
        self.data_ids = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        print(f"Found {len(self.data_ids)} images in {self.image_dir}")

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        
        if self.has_masks:
            mask_path = os.path.join(self.masks_dir, data_id)
            mask = Image.open(mask_path)
            mask = convert_mask(mask)
            
            if self.transform:
                image = self.transform(image)
                mask = self.mask_transform(mask) * 255
            
            return image, mask, data_id
        else:
            # Test mode: no masks
            if self.transform:
                image = self.transform(image)
            
            return image, None, data_id


# ============================================================================
# Model: Segmentation Head (ConvNeXt-style) - Must match training
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

    return np.nanmean(iou_per_class), iou_per_class


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

    return np.mean(dice_per_class), dice_per_class


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


# ============================================================================
# Visualization Functions
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    """Save a side-by-side comparison of input, ground truth, and prediction."""
    # Denormalize image
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    # Convert masks to color
    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(gt_color)
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle(f'Sample: {data_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_test_prediction_visual(img_tensor, pred_mask, output_path, data_id):
    """Save visualization for test images (no ground truth)."""
    # Denormalize image
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    # Convert prediction to color
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img)
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(pred_color)
    axes[1].set_title('Predicted Segmentation', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.suptitle(f'Test Sample: {data_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_summary(results, output_dir, has_masks=True):
    """Save metrics summary to a text file and create bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    if not has_masks:
        # Test mode: no metrics to save
        print("\nTest mode: No ground truth available, skipping metrics calculation.")
        return

    # Save text summary
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 25 + "EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Overall Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write(f"  Mean Dice:         {results['mean_dice']:.4f}\n")
        f.write(f"  Mean Pixel Acc:    {results['mean_pixel_acc']:.4f}\n")
        f.write(f"  Avg Inference Time: {results['avg_inference_time']:.2f} ms/image\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Per-Class IoU:\n")
        f.write("-" * 80 + "\n")
        for i, (name, iou) in enumerate(zip(class_names, results['class_iou'])):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {iou_str}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("Per-Class Dice Score:\n")
        f.write("-" * 80 + "\n")
        for i, (name, dice) in enumerate(zip(class_names, results['class_dice'])):
            dice_str = f"{dice:.4f}" if not np.isnan(dice) else "N/A"
            f.write(f"  {name:<20}: {dice_str}\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Saved evaluation metrics to '{filepath}'")

    # Create bar chart for per-class IoU
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # IoU chart
    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    bars1 = ax1.bar(range(n_classes), valid_iou, 
                    color=[color_palette[i] / 255 for i in range(n_classes)],
                    edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(n_classes))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.set_ylabel('IoU Score', fontsize=12)
    ax1.set_title(f'Per-Class IoU (Mean: {results["mean_iou"]:.4f})', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=results['mean_iou'], color='red', linestyle='--', linewidth=2, label=f'Mean: {results["mean_iou"]:.3f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Dice chart
    valid_dice = [dice if not np.isnan(dice) else 0 for dice in results['class_dice']]
    bars2 = ax2.bar(range(n_classes), valid_dice,
                    color=[color_palette[i] / 255 for i in range(n_classes)],
                    edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(n_classes))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.set_title(f'Per-Class Dice Score (Mean: {results["mean_dice"]:.4f})', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=results['mean_dice'], color='red', linestyle='--', linewidth=2, label=f'Mean: {results["mean_dice"]:.3f}')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved per-class metrics chart to '{output_dir}/per_class_metrics.png'")


# ============================================================================
# Main Validation/Test Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Segmentation validation/test script')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to validation/test dataset')
    parser.add_argument('--output_dir', type=str, default='/content/drive/MyDrive/SWOC26_Training/results',
                        help='Directory to save prediction visualizations')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of comparison visualizations to save')
    parser.add_argument('--test_mode', action='store_true',
                        help='Test mode (no ground truth masks available)')
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    mode_str = "TEST" if args.test_mode else "VALIDATION"
    print(" " * 30 + f"{mode_str} MODE")
    print("=" * 80)
    print(f"\n🔧 Configuration:")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")

    # Image dimensions (must match training)
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)
    print(f"  Image size: {h}x{w}")

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

    # Create dataset
    print(f"\n📁 Loading dataset...")
    dataset = MaskDataset(
        data_dir=args.data_dir, 
        transform=transform, 
        mask_transform=mask_transform,
        has_masks=not args.test_mode
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"  Loaded {len(dataset)} samples")
    print(f"  Batches: {len(data_loader)}")

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

    # Get embedding dimension
    sample_img, _, _ = dataset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(sample_img)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"  Embedding dimension: {n_embedding}")

    # Load classifier
    print(f"\n🏗️  Loading segmentation head...")
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    )
    classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()
    print(f"  ✓ Model loaded successfully!")

    # Create subdirectories for outputs
    masks_dir = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    visualizations_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    # Run inference
    print(f"\n" + "=" * 80)
    print(" " * 28 + "RUNNING INFERENCE")
    print("=" * 80 + "\n")

    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    all_class_iou = []
    all_class_dice = []
    inference_times = []
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Processing", unit="batch", ncols=100)
        for batch_idx, batch_data in enumerate(pbar):
            if args.test_mode:
                imgs, _, data_ids = batch_data
                labels = None
            else:
                imgs, labels, data_ids = batch_data
                labels = labels.to(device)

            imgs = imgs.to(device)

            # Measure inference time
            start_time = time.time()
            
            # Forward pass
            output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(output.to(device))
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            predicted_masks = torch.argmax(outputs, dim=1)
            
            inference_time = (time.time() - start_time) * 1000 / imgs.shape[0]  # ms per image
            inference_times.append(inference_time)

            # Calculate metrics (only if ground truth available)
            if not args.test_mode:
                labels_squeezed = labels.squeeze(dim=1).long()
                iou, class_iou = compute_iou(outputs, labels_squeezed, num_classes=n_classes)
                dice, class_dice = compute_dice(outputs, labels_squeezed, num_classes=n_classes)
                pixel_acc = compute_pixel_accuracy(outputs, labels_squeezed)

                iou_scores.append(iou)
                dice_scores.append(dice)
                pixel_accuracies.append(pixel_acc)
                all_class_iou.append(class_iou)
                all_class_dice.append(class_dice)
                
                pbar.set_postfix(iou=f"{iou:.3f}", inference_ms=f"{inference_time:.1f}")
            else:
                pbar.set_postfix(inference_ms=f"{inference_time:.1f}")

            # Save predictions for every image
            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                # Save raw prediction mask (class IDs 0-9)
                pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)
                pred_img = Image.fromarray(pred_mask)
                pred_img.save(os.path.join(masks_dir, f'{base_name}_pred.png'))

                # Save colored prediction mask (RGB visualization)
                pred_color = mask_to_color(pred_mask)
                cv2.imwrite(os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                            cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # Save visualization for first N samples
                if sample_count < args.num_samples:
                    if args.test_mode:
                        save_test_prediction_visual(
                            imgs[i], predicted_masks[i],
                            os.path.join(visualizations_dir, f'sample_{sample_count:03d}_{base_name}.png'),
                            data_id
                        )
                    else:
                        save_prediction_comparison(
                            imgs[i], labels_squeezed[i], predicted_masks[i],
                            os.path.join(visualizations_dir, f'sample_{sample_count:03d}_{base_name}.png'),
                            data_id
                        )

                sample_count += 1

    # Aggregate results
    print(f"\n{'='*80}")
    print(" " * 35 + "RESULTS")
    print(f"{'='*80}\n")
    
    avg_inference_time = np.mean(inference_times)
    print(f"  ⏱️  Average Inference Time: {avg_inference_time:.2f} ms/image")
    print(f"  📊 Total Images Processed: {len(dataset)}")
    
    if not args.test_mode:
        mean_iou = np.nanmean(iou_scores)
        mean_dice = np.nanmean(dice_scores)
        mean_pixel_acc = np.mean(pixel_accuracies)
        avg_class_iou = np.nanmean(all_class_iou, axis=0)
        avg_class_dice = np.nanmean(all_class_dice, axis=0)

        results = {
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'mean_pixel_acc': mean_pixel_acc,
            'class_iou': avg_class_iou,
            'class_dice': avg_class_dice,
            'avg_inference_time': avg_inference_time
        }

        print(f"\n  📈 Evaluation Metrics:")
        print(f"  {'─'*76}")
        print(f"  Mean IoU:           {mean_iou:.4f}")
        print(f"  Mean Dice Score:    {mean_dice:.4f}")
        print(f"  Mean Pixel Accuracy:{mean_pixel_acc:.4f}")
        print(f"  {'─'*76}")

        # Save metrics
        save_metrics_summary(results, args.output_dir, has_masks=True)
    
    print(f"\n{'='*80}")
    print(" " * 32 + "COMPLETE!")
    print(f"{'='*80}\n")
    
    print(f"  💾 All outputs saved to: {args.output_dir}/")
    print(f"     - masks/            : Raw prediction masks (class IDs 0-9)")
    print(f"     - masks_color/      : Colored prediction masks (RGB)")
    print(f"     - visualizations/   : Comparison images ({min(args.num_samples, len(dataset))} samples)")
    if not args.test_mode:
        print(f"     - evaluation_metrics.txt")
        print(f"     - per_class_metrics.png")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
