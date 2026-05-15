import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

from recetas.config import DATA, MODELS, FRUITS360_DIR

FRUITS_DIR  = FRUITS360_DIR
CATALOG     = DATA.ingredient_catalog
OUT_WEIGHTS = MODELS.cnn_weights
OUT_LABELS  = MODELS.class_labels


class CatalogDataset(Dataset):
    """ImageFolder restricted to the ingredient classes present in the catalog."""

    def __init__(self, fruits_train_dir: Path, catalog: dict, transform=None):
        ingr_to_cls: dict[str, str] = {}
        for ingr, path in catalog.items():
            if path:
                cls = Path(path).parent.name
                ingr_to_cls[ingr] = cls

        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = sorted(ingr_to_cls.keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

        for ingr in self.classes:
            fruits_cls = ingr_to_cls[ingr]
            cls_dir = fruits_train_dir / fruits_cls
            if not cls_dir.exists():
                continue
            label = self.class_to_idx[ingr]
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((img_path, label))

        print(f"Dataset: {len(self.classes)} classes | {len(self.samples):,} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def build_model(num_classes: int, device):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model.to(device)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        correct += (out.argmax(1) == labels).sum().item()
        total += len(imgs)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * len(imgs)
        correct += (out.argmax(1) == labels).sum().item()
        total += len(imgs)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val-split",  type=float, default=0.15)
    parser.add_argument("--workers",    type=int,   default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(CATALOG) as f:
        catalog = json.load(f)

    train_tf, val_tf = get_transforms()
    train_dir = FRUITS_DIR / "Training"

    full_ds = CatalogDataset(train_dir, catalog, transform=train_tf)
    n_val   = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    class TransformSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            img = Image.open(path).convert("RGB")
            return self.transform(img), label

    val_ds = TransformSubset(val_ds, val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    num_classes = len(full_ds.classes)
    print(f"\nClasses: {num_classes} | Train: {n_train:,} | Val: {n_val:,}")

    model     = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    print(f"\n{'Epoch':>5} {'Train Loss':>10} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>10} {'LR':>8}")
    print("─" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"{epoch:>5} {tr_loss:>10.4f} {tr_acc:>9.1%} {va_loss:>10.4f} {va_acc:>9.1%} {lr:>8.2e}  ({elapsed:.0f}s)")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), OUT_WEIGHTS)
            print(f"        saved  val_acc={va_acc:.1%}")

    class_labels = {str(i): name for i, name in enumerate(full_ds.classes)}
    with open(OUT_LABELS, "w") as f:
        json.dump(class_labels, f, indent=2)

    print(f"\nDone. best_val_acc={best_val_acc:.1%}")
    print(f"  weights: {OUT_WEIGHTS}")
    print(f"  labels:  {OUT_LABELS}  ({num_classes} classes)")


if __name__ == "__main__":
    main()
