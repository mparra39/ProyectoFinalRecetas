"""
train_extended_cnn.py — Reentrenamiento del clasificador de ingredientes.

Combina 4 datasets:
  1. Fruits-360          (moltean/fruits)                       — ya en caché
  2. Vegetable Images    (misrakahmed/vegetable-image-dataset)  — descarga ~300MB
  3. Fruit & Veg Recog.  (kritikseth/fruit-and-vegetable-image-recognition) — ~50MB
  4. Recipe Ingredients  (fasihcs/recipe-ingredients-image-dataset) — ya en caché

Ejecución:
    conda run -n recipes python recetas/modeling/train_extended_cnn.py

Opciones:
    --epochs     20        (número de épocas)
    --batch-size 256       (optimizado para RTX 4070 8GB)
    --workers    12        (DataLoader workers)
    --lr         1e-3
    --no-compile           (desactivar torch.compile si da problemas)
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import kagglehub
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
from recetas.config import MODELS, DATA

OUT_WEIGHTS  = MODELS.cnn_weights
OUT_LABELS   = MODELS.class_labels
OUT_CATALOG  = DATA.ingredient_catalog

MIN_IMAGES   = 30    # descartar clases con < N imágenes
MAX_IMAGES   = 800   # cap para balancear clases muy grandes
VAL_SPLIT    = 0.15
IMAGE_EXTS   = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZACIÓN DE NOMBRES
# ─────────────────────────────────────────────────────────────────────────────
def normalize_class(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r'\d+', '', name)
    name = re.sub(r'[-_]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA Y COLECCIÓN DE DATASETS
# ─────────────────────────────────────────────────────────────────────────────
def collect_dir(root: Path, sources: dict[str, list[Path]], tag: str) -> int:
    """Recorre subcarpetas de root y acumula imágenes por clase normalizada."""
    n_classes = 0
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = sorted(f for f in cls_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        if not imgs:
            continue
        key = normalize_class(cls_dir.name)
        sources[key].extend(imgs)
        n_classes += 1
    print(f"  [{tag}] {n_classes} clases desde {root}")
    return n_classes


def download_datasets() -> dict[str, list[Path]]:
    """Descarga los 4 datasets y devuelve {clase_norm: [rutas_imagen]}."""
    sources: dict[str, list[Path]] = defaultdict(list)

    # ── 1. Fruits-360 ─────────────────────────────────────────────────────────
    print("\n[1/4] Fruits-360 (ya en caché)…")
    path = Path(kagglehub.dataset_download("moltean/fruits"))
    train_dir = next(path.rglob("Training"), None)
    if train_dir:
        collect_dir(train_dir, sources, "Fruits-360")

    # ── 2. Vegetable Image Dataset ────────────────────────────────────────────
    print("\n[2/4] Vegetable Image Dataset (descargando si hace falta)…")
    path = Path(kagglehub.dataset_download("misrakahmed/vegetable-image-dataset"))
    # Estructura: .../Vegetable Images/train/<Clase>/
    train_dir = next(path.rglob("train"), None)
    if train_dir and train_dir.is_dir():
        collect_dir(train_dir, sources, "Vegetable")
    else:
        # Fallback: buscar cualquier directorio con subcarpetas de imágenes
        for d in sorted(path.rglob("*")):
            if d.is_dir() and any(f.suffix.lower() in IMAGE_EXTS
                                  for f in d.iterdir() if f.is_file()):
                parent = d.parent
                collect_dir(parent, sources, "Vegetable")
                break

    # ── 3. Fruit & Veg Recognition ────────────────────────────────────────────
    print("\n[3/4] Fruit & Veg Recognition (descargando si hace falta)…")
    path = Path(kagglehub.dataset_download(
        "kritikseth/fruit-and-vegetable-image-recognition"))
    train_dir = next(path.rglob("train"), None)
    if train_dir and train_dir.is_dir():
        collect_dir(train_dir, sources, "FVR")

    # ── 4. Recipe Ingredients ─────────────────────────────────────────────────
    print("\n[4/4] Recipe Ingredients (ya en caché)…")
    path = Path(kagglehub.dataset_download("fasihcs/recipe-ingredients-image-dataset"))
    for d in sorted(path.rglob("*")):
        if not d.is_dir():
            continue
        imgs = sorted(f for f in d.iterdir()
                      if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        if imgs:
            key = normalize_class(d.name)
            sources[key].extend(imgs)

    print(f"\nClases únicas brutas: {len(sources)}")
    print(f"Imágenes brutas totales: {sum(len(v) for v in sources.values()):,}")
    return sources


# ─────────────────────────────────────────────────────────────────────────────
# FILTRADO Y BALANCEO
# ─────────────────────────────────────────────────────────────────────────────
def build_filtered(sources: dict[str, list[Path]]) -> dict[str, list[Path]]:
    rng = np.random.default_rng(42)
    filtered: dict[str, list[Path]] = {}
    for cls, imgs in sources.items():
        imgs_unique = list(dict.fromkeys(imgs))   # deduplicar manteniendo orden
        if len(imgs_unique) < MIN_IMAGES:
            continue
        if len(imgs_unique) > MAX_IMAGES:
            idxs = rng.choice(len(imgs_unique), MAX_IMAGES, replace=False)
            imgs_unique = [imgs_unique[i] for i in sorted(idxs)]
        filtered[cls] = imgs_unique

    print(f"\nTras filtro (≥{MIN_IMAGES} imgs, ≤{MAX_IMAGES} imgs):")
    print(f"  Clases: {len(filtered)}")
    print(f"  Imágenes: {sum(len(v) for v in filtered.values()):,}")
    sizes = sorted(len(v) for v in filtered.values())
    print(f"  Min/Mediana/Max por clase: {min(sizes)} / {int(np.median(sizes))} / {max(sizes)}")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class IngredientDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.55, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.1),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf


def build_loaders(filtered: dict[str, list[Path]],
                  class_list: list[str],
                  class_to_idx: dict[str, int],
                  batch_size: int,
                  workers: int):
    train_tf, val_tf = get_transforms()

    all_samples: list[tuple[Path, int]] = []
    for cls in class_list:
        label = class_to_idx[cls]
        for p in filtered[cls]:
            all_samples.append((p, label))

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(all_samples)).tolist()
    all_samples = [all_samples[i] for i in perm]

    n_val   = int(len(all_samples) * VAL_SPLIT)
    n_train = len(all_samples) - n_val
    train_s = all_samples[:n_train]
    val_s   = all_samples[n_train:]

    # WeightedRandomSampler para balancear durante el entrenamiento
    class_counts: dict[int, int] = defaultdict(int)
    for _, label in train_s:
        class_counts[label] += 1
    weights = [1.0 / class_counts[label] for _, label in train_s]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_s), replacement=True)

    train_ds = IngredientDataset(train_s, transform=train_tf)
    val_ds   = IngredientDataset(val_s,   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=workers, pin_memory=True,
                              persistent_workers=workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True,
                              persistent_workers=workers > 0)

    print(f"\nTrain: {len(train_ds):,} imágenes ({len(train_loader)} batches)")
    print(f"Val:   {len(val_ds):,} imágenes  ({len(val_loader)} batches)")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# MODELO
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, device) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Dispositivo : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1024**3
        print(f"GPU         : {props.name}  ({vram:.1f} GB VRAM)")
    print(f"{'='*60}")

    # 1. Datasets
    sources  = download_datasets()
    filtered = build_filtered(sources)

    class_list    = sorted(filtered.keys())
    class_to_idx  = {c: i for i, c in enumerate(class_list)}
    num_classes   = len(class_list)

    print(f"\nClases finales: {num_classes}")

    # 2. Guardar metadatos
    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    OUT_CATALOG.parent.mkdir(parents=True, exist_ok=True)

    class_labels = {str(i): name for i, name in enumerate(class_list)}
    with open(OUT_LABELS, "w") as f:
        json.dump(class_labels, f, indent=2, ensure_ascii=False)
    print(f"class_labels.json → {OUT_LABELS}")

    # Catálogo: preferir imágenes de datasets con fondo real
    ingredient_catalog: dict[str, str] = {}
    for cls in class_list:
        imgs = filtered[cls]
        real_bg = [p for p in imgs if
                   "misrakahmed" in str(p) or "kritikseth" in str(p)]
        ingredient_catalog[cls] = str(real_bg[0] if real_bg else imgs[0])

    with open(OUT_CATALOG, "w") as f:
        json.dump(ingredient_catalog, f, indent=2, ensure_ascii=False)
    print(f"ingredient_catalog.json → {OUT_CATALOG}")

    # 3. DataLoaders
    train_loader, val_loader = build_loaders(
        filtered, class_list, class_to_idx, args.batch_size, args.workers)

    # 4. Modelo
    model = build_model(num_classes, device)
    if args.compile and device.type == "cuda":
        print("Compilando modelo con torch.compile…")
        model = torch.compile(model)

    total   = sum(p.numel() for p in model.parameters())
    trainab = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros: {total:,} total | {trainab:,} entrenables")

    # 5. Optimizador + scheduler + AMP
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    warmup_epochs = min(5, args.epochs // 4)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 6. Loop
    best_val_acc = 0.0
    OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()

    print(f"\n{'Época':>5} {'Loss Tr':>9} {'Acc Tr':>8} {'Loss Val':>9} "
          f"{'Acc Val':>8} {'Top-5 Val':>10} {'LR':>9} {'Tiempo':>7}")
    print("─" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            tr_loss    += loss.item() * len(imgs)
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total   += len(imgs)

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        va_loss, va_correct, va_top5, va_total = 0.0, 0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                va_loss    += loss.item() * len(imgs)
                va_correct += (logits.argmax(1) == labels).sum().item()
                top5        = logits.topk(5, dim=1).indices
                va_top5    += (labels.unsqueeze(1) == top5).any(dim=1).sum().item()
                va_total   += len(imgs)

        scheduler.step()
        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        tr_acc  = tr_correct / tr_total
        va_acc  = va_correct / va_total
        va_top5_acc = va_top5 / va_total

        saved = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            # Guardar estado sin torch.compile wrapper
            state = (model._orig_mod.state_dict()
                     if hasattr(model, "_orig_mod") else model.state_dict())
            torch.save(state, OUT_WEIGHTS)
            saved = " ← guardado"

        print(f"{epoch:>5} {tr_loss/tr_total:>9.4f} {tr_acc:>7.1%} "
              f"{va_loss/va_total:>9.4f} {va_acc:>7.1%} {va_top5_acc:>9.1%} "
              f"{lr:>9.2e} {elapsed:>5.0f}s{saved}")

    print(f"\n{'='*60}")
    print(f"Entrenamiento completo.")
    print(f"Mejor val_acc : {best_val_acc:.2%}")
    print(f"Clases        : {num_classes}")
    print(f"Pesos         : {OUT_WEIGHTS}")
    print(f"{'='*60}")

    # 7. Subir a Hugging Face
    if args.push:
        _push_to_hub(args.hf_token, args.hf_username, num_classes, best_val_acc)


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD A HUGGING FACE HUB
# ─────────────────────────────────────────────────────────────────────────────
def _push_to_hub(token: str, username: str, num_classes: int, best_acc: float):
    from huggingface_hub import login
    login(token=token)
    api  = HfApi()
    repo = f"{username}/recipe-ingredient-classifier"
    api.create_repo(repo_id=repo, exist_ok=True, repo_type="model")
    print(f"\nSubiendo a https://huggingface.co/{repo}…")
    for fpath, name in [
        (OUT_WEIGHTS,  "efficientnet_ingredients.pth"),
        (OUT_LABELS,   "class_labels.json"),
        (OUT_CATALOG,  "ingredient_catalog.json"),
    ]:
        api.upload_file(path_or_fileobj=str(fpath),
                        path_in_repo=name,
                        repo_id=repo)
        print(f"  {name}")

    # README con metadatos
    readme = (
        f"# Recipe Ingredient Classifier\n\n"
        f"EfficientNet-B0 trained on {num_classes} ingredient classes.\n"
        f"Best val accuracy: {best_acc:.2%}\n\n"
        f"## Datasets\n"
        f"- Fruits-360 (moltean/fruits)\n"
        f"- Vegetable Image Dataset (misrakahmed/vegetable-image-dataset)\n"
        f"- Fruit & Veg Recognition (kritikseth/fruit-and-vegetable-image-recognition)\n"
        f"- Recipe Ingredients (fasihcs/recipe-ingredients-image-dataset)\n"
    )
    readme_path = OUT_WEIGHTS.parent / "README_cnn.md"
    readme_path.write_text(readme)
    api.upload_file(path_or_fileobj=str(readme_path),
                    path_in_repo="README.md",
                    repo_id=repo)
    print(f"  README.md")
    print(f"\nSubida completa → {repo}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Entrenar CNN de ingredientes extendido")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch-size",   type=int,   default=128,
                   help="128 seguro para RTX 4070 8GB con EfficientNet-B0")
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--workers",      type=int,   default=12,
                   help="DataLoader workers (12 recomendado para 22 núcleos)")
    p.add_argument("--no-compile",   dest="compile", action="store_false",
                   help="Desactivar torch.compile")
    p.add_argument("--push",         action="store_true",
                   help="Subir pesos a Hugging Face Hub al terminar")
    p.add_argument("--hf-token",     type=str,   default="",
                   help="Token de Hugging Face (requiere --push)")
    p.add_argument("--hf-username",  type=str,   default="ramonsj11")
    p.set_defaults(compile=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
