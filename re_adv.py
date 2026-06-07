# -*- coding: utf-8 -*-
"""Billboard Detector — Setup, Dataset Preparation & Training (with Metrics)"""

# ============================================================
# BILLBOARD DETECTOR — PART 1: SETUP & DATASET PREPARATION
# ============================================================

# ── CELL 1: Install & verify environment ────────────────────
!pip install -q ultralytics supervision

import os, cv2, torch, yaml, zipfile, shutil, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from google.colab import files

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detected. Go to Runtime → Change runtime type → T4 GPU")

# ── CELL 2: Upload your dataset zip ─────────────────────────
print("Upload your dataset .zip file:")
uploaded = files.upload()

zip_path = next((f for f in uploaded if f.endswith('.zip')), None)
if zip_path is None:
    raise FileNotFoundError("❌ No .zip file found. Please upload your dataset zip.")

print(f"✅ Received: {zip_path}")

# ── CELL 3: Extract and inspect ─────────────────────────────
RAW  = "datasets/raw"
OUT  = "datasets/billboard"

!rm -rf datasets && mkdir -p datasets
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(RAW)

print("\n📁 Extracted structure (first 30 files):")
!find datasets/raw -type f | head -30

# ── CELL 4: Auto-locate images & annotations folders ────────
img_dir = None
ann_dir = None

for root, dirs, _ in os.walk(RAW):
    for d in dirs:
        full = os.path.join(root, d)
        if d == "images" and img_dir is None:
            img_dir = full
        elif d in ("annotations", "labels", "csvs") and ann_dir is None:
            ann_dir = full

# Fallback: scan for CSVs anywhere
if ann_dir is None:
    for root, dirs, files_list in os.walk(RAW):
        csvs = [f for f in files_list if f.endswith('.csv')]
        if csvs:
            ann_dir = root
            break

if img_dir is None or ann_dir is None:
    print("❌ Could not auto-find folders. Listing raw structure:")
    !find datasets/raw -type d
    raise FileNotFoundError(
        "Set img_dir and ann_dir manually above after inspecting the structure."
    )

print(f"✅ Images      : {img_dir}  ({len(os.listdir(img_dir))} files)")
print(f"✅ Annotations : {ann_dir}  ({len(os.listdir(ann_dir))} files)")

# ── CELL 5: Inspect one annotation CSV ──────────────────────
csv_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.csv')])
print(f"\nTotal annotation CSVs: {len(csv_files)}")
print("\nSample CSV columns:")
sample_df = pd.read_csv(os.path.join(ann_dir, csv_files[0]))
print(sample_df.head())
print(f"\nColumns: {list(sample_df.columns)}")
print(f"Labels found in sample: {sample_df['label'].unique()}")

# ── CELL 6: Build class map & verify coordinate format ──────
all_labels = set()
coord_stats = {'max_x': 0, 'max_y': 0, 'min_x': 9999, 'min_y': 9999}

for cf in csv_files:
    df = pd.read_csv(os.path.join(ann_dir, cf))
    all_labels.update(df['label'].unique())
    # Detect if coords are normalized (0-1) or pixel values
    for col in ['x1', 'x2']:
        if col in df.columns:
            coord_stats['max_x'] = max(coord_stats['max_x'], df[col].max())
            coord_stats['min_x'] = min(coord_stats['min_x'], df[col].min())
    for col in ['y1', 'y2']:
        if col in df.columns:
            coord_stats['max_y'] = max(coord_stats['max_y'], df[col].max())
            coord_stats['min_y'] = min(coord_stats['min_y'], df[col].min())

class_names = sorted(all_labels)
class_to_id = {name: i for i, name in enumerate(class_names)}

IS_NORMALIZED = (coord_stats['max_x'] <= 1.0 and coord_stats['max_y'] <= 1.0)
print(f"\nClasses ({len(class_names)}): {class_names}")
print(f"Coordinate range — X: [{coord_stats['min_x']:.3f}, {coord_stats['max_x']:.3f}]")
print(f"Coordinate range — Y: [{coord_stats['min_y']:.3f}, {coord_stats['max_y']:.3f}]")
print(f"Coords already normalized: {IS_NORMALIZED}")
# If IS_NORMALIZED=False → pixel coords (need image W/H to normalize)

# ── CELL 7: Match image ↔ annotation pairs ──────────────────
VALID_EXT = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

base_names = []
unmatched = []
for cf in csv_files:
    stem = os.path.splitext(cf)[0]
    found = False
    for ext in VALID_EXT:
        img_path = os.path.join(img_dir, stem + ext)
        if os.path.exists(img_path):
            base_names.append((stem, ext))
            found = True
            break
    if not found:
        unmatched.append(stem)

print(f"\n✅ Matched pairs      : {len(base_names)}")
print(f"⚠️  Unmatched CSVs    : {len(unmatched)}")
if unmatched[:5]:
    print(f"   Samples           : {unmatched[:5]}")

if len(base_names) == 0:
    raise ValueError("No matched image-annotation pairs found. Check folder structure.")

# ── CELL 8: Convert to YOLO format with normalization ───────
random.seed(42)
random.shuffle(base_names)

n       = len(base_names)
n_train = int(n * 0.80)
n_val   = int(n * 0.10)
# remaining 10% → val overflow (no separate test split needed)

splits = {
    'train': base_names[:n_train],
    'val'  : base_names[n_train : n_train + n_val],
}

skipped = 0
for split_name, items in splits.items():
    os.makedirs(f"{OUT}/{split_name}/images", exist_ok=True)
    os.makedirs(f"{OUT}/{split_name}/labels", exist_ok=True)

    for stem, ext in items:
        src_img = os.path.join(img_dir, stem + ext)
        dst_img = os.path.join(OUT, split_name, "images", stem + ext)
        dst_lbl = os.path.join(OUT, split_name, "labels", stem + ".txt")

        # Read image dimensions for normalization
        img = cv2.imread(src_img)
        if img is None:
            skipped += 1
            continue
        H, W = img.shape[:2]

        df = pd.read_csv(os.path.join(ann_dir, stem + '.csv'))
        lines = []
        for _, row in df.iterrows():
            x1, y1 = float(row['x1']), float(row['y1'])
            x2, y2 = float(row['x2']), float(row['y2'])

            # Normalize if pixel coords
            if not IS_NORMALIZED:
                x1, x2 = x1 / W, x2 / W
                y1, y2 = y1 / H, y2 / H

            # Clamp to [0, 1]
            x1 = max(0.0, min(1.0, x1))
            x2 = max(0.0, min(1.0, x2))
            y1 = max(0.0, min(1.0, y1))
            y2 = max(0.0, min(1.0, y2))

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = abs(x2 - x1)
            bh = abs(y2 - y1)

            # Skip degenerate boxes
            if bw < 0.005 or bh < 0.005:
                continue

            cid = class_to_id[row['label']]
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if lines:
            shutil.copy2(src_img, dst_img)
            with open(dst_lbl, 'w') as f:
                f.write('\n'.join(lines) + '\n')
        else:
            skipped += 1  # empty annotation

    print(f"  {split_name:6s}: {len(items)} images")

print(f"\n⚠️  Skipped (bad/empty): {skipped}")

# ── CELL 9: Write data.yaml ──────────────────────────────────
data_yaml = {
    'path'  : os.path.abspath(OUT),
    'train' : 'train/images',
    'val'   : 'val/images',
    'nc'    : len(class_names),
    'names' : class_names,
}
yaml_path = f"{OUT}/data.yaml"
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"\n✅ data.yaml saved → {yaml_path}")
print(yaml.dump(data_yaml))

# ── CELL 10: Visualize samples with ground-truth boxes ───────
def show_yolo_samples(split='train', n_samples=6):
    img_folder = f"{OUT}/{split}/images"
    lbl_folder = f"{OUT}/{split}/labels"
    imgs = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(imgs)
    imgs = imgs[:n_samples]

    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes = np.array(axes).flatten()

    colors = plt.cm.get_cmap('tab10', len(class_names))

    for ax, fname in zip(axes, imgs):
        img_path = os.path.join(img_folder, fname)
        lbl_path = os.path.join(lbl_folder, os.path.splitext(fname)[0] + '.txt')

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        ax.imshow(img)

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                    x1 = (cx - bw / 2) * W
                    y1 = (cy - bh / 2) * H
                    rect = patches.Rectangle(
                        (x1, y1), bw * W, bh * H,
                        linewidth=2, edgecolor=colors(cid), facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 4, class_names[cid], color=colors(cid),
                            fontsize=9, weight='bold',
                            bbox=dict(fc='white', alpha=0.5, pad=1))
        ax.axis('off')
        ax.set_title(fname[:30], fontsize=8)

    for ax in axes[len(imgs):]:
        ax.axis('off')

    plt.suptitle(f"Ground Truth — {split} split", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()

show_yolo_samples('train', n_samples=6)

print("\n✅ PART 1 COMPLETE — Dataset is ready.")
print(f"   Train : {len(splits['train'])} images")
print(f"   Val   : {len(splits['val'])} images")
print(f"\n▶  Proceeding to training...")


# ============================================================
# BILLBOARD DETECTOR — PART 2: TRAINING
# ============================================================

# ── CELL 1: Reinstall & reload ───────────────────────────────
!pip install -q ultralytics

import os, yaml, glob, shutil, torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

OUT  = "datasets/billboard"
YAML = f"{OUT}/data.yaml"

# ── CELL 2: Verify dataset ────────────────────────────────────
if not os.path.exists(YAML):
    raise FileNotFoundError("❌ data.yaml not found — run Part 1 first.")

with open(YAML) as f:
    cfg = yaml.safe_load(f)

print(f"✅ Dataset loaded — Classes: {cfg['names']}")
for split in ['train', 'val']:
    img_dir = os.path.join(cfg['path'], split, 'images')
    n = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
    print(f"   {split:6s}: {n} images")

# ── CELL 3: Auto-select model size ───────────────────────────
n_train = len(os.listdir(os.path.join(cfg['path'], 'train', 'images')))

if n_train < 500:
    MODEL_SIZE, BATCH = "yolov8s.pt", 16
elif n_train < 2000:
    MODEL_SIZE, BATCH = "yolov8m.pt", 12
else:
    MODEL_SIZE, BATCH = "yolov8l.pt", 8

print(f"\n📊 Train size: {n_train}  →  model={MODEL_SIZE}  batch={BATCH}")
model = YOLO(MODEL_SIZE)

# ── CELL 4: Train ────────────────────────────────────────────
results = model.train(
    data          = YAML,
    epochs        = 200,
    imgsz         = 640,
    batch         = BATCH,
    lr0           = 0.005,
    lrf           = 0.01,
    cos_lr        = True,
    warmup_epochs = 5,
    warmup_bias_lr= 0.1,
    warmup_momentum=0.8,
    weight_decay  = 0.0005,
    optimizer     = "AdamW",
    momentum      = 0.937,
    box           = 7.5,
    cls           = 0.5,
    dfl           = 1.5,
    hsv_h         = 0.015,
    hsv_s         = 0.7,
    hsv_v         = 0.4,
    degrees       = 10.0,
    translate     = 0.1,
    scale         = 0.6,
    shear         = 5.0,
    perspective   = 0.0005,
    flipud        = 0.0,
    fliplr        = 0.5,
    mosaic        = 1.0,
    mixup         = 0.1,
    copy_paste    = 0.0,
    close_mosaic  = 20,
    iou           = 0.6,
    patience      = 40,
    project       = "runs/train",
    name          = "billboard_v1",
    exist_ok      = False,
    seed          = 42,
    verbose       = True,
    save          = True,
    save_period   = 10,
    plots         = True,
    overlap_mask  = True,
    val           = True,
    cache         = False,
)

print("\n✅ Training complete!")

# ── CELL 5: Robust weight finder ─────────────────────────────
def find_run_dir(name="billboard_v1"):
    """Search all possible Ultralytics output locations."""
    patterns = [
        f"/content/runs/train/{name}*/weights/best.pt",
        f"/content/runs/detect/runs/train/{name}*/weights/best.pt",
        f"/content/runs/detect/{name}*/weights/best.pt",
        f"/content/**/weights/best.pt",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            best_pt  = hits[-1]
            run_dir  = os.path.dirname(os.path.dirname(best_pt))
            print(f"✅ Found best.pt : {best_pt}")
            print(f"   Run dir       : {run_dir}")
            return run_dir, best_pt
    return None, None

run_dir, best_pt = find_run_dir("billboard_v1")

if best_pt is None:
    if hasattr(results, 'save_dir'):
        run_dir  = str(results.save_dir)
        best_pt  = os.path.join(run_dir, "weights", "best.pt")
        print(f"✅ Using results.save_dir: {run_dir}")
    else:
        print("❌ Could not locate weights. Listing /content/runs:")
        !find /content/runs -name "*.pt" 2>/dev/null
        raise FileNotFoundError("Cannot find best.pt. See listing above.")

# ── CELL 6: Training curves ───────────────────────────────────
def show_training_curves(run_dir):
    import pandas as pd
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️  results.csv not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(f"\nColumns in results.csv: {list(df.columns)}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    loss_pairs = [
        ('train/box_loss', 'val/box_loss',  'Box Loss'),
        ('train/cls_loss', 'val/cls_loss',  'Class Loss'),
        ('train/dfl_loss', 'val/dfl_loss',  'DFL Loss'),
    ]
    metric_cols = [
        ('metrics/mAP50(B)',     'mAP@0.5'),
        ('metrics/mAP50-95(B)', 'mAP@0.5:0.95'),
        ('metrics/precision(B)','Precision'),
    ]

    for ax, (tr, va, title) in zip(axes[0], loss_pairs):
        if tr in df.columns:
            ax.plot(df[tr], label='train', color='royalblue')
        if va in df.columns:
            ax.plot(df[va], label='val',   color='tomato')
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(True, alpha=0.3)

    for ax, (col, title) in zip(axes[1], metric_cols):
        if col in df.columns:
            ax.plot(df[col], color='seagreen')
            if col == 'metrics/mAP50(B)':
                best_ep = df[col].idxmax()
                ax.axvline(best_ep, color='red', linestyle='--', alpha=0.7,
                           label=f'Best: {df[col].max():.4f} @ep{best_ep}')
                ax.legend()
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)

    plt.suptitle("Billboard Detector — Training Curves", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()

    print(f"\n📊 Best metrics:")
    for col in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                'metrics/precision(B)', 'metrics/recall(B)']:
        if col in df.columns:
            print(f"   {col:35s} = {df[col].max():.4f}")

show_training_curves(run_dir)

# ── CELL 7: Validation metrics on val split ───────────────────
print("\n📊 Validating on val split using best.pt ...")
best_model = YOLO(best_pt)
val_metrics = best_model.val(
    data    = YAML,
    split   = 'val',
    conf    = 0.001,
    iou     = 0.6,
    imgsz   = 640,
    verbose = True,
)

class_names_model = list(best_model.names.values())

print(f"\n{'═'*45}")
print(f"  VAL  mAP@0.5        : {val_metrics.box.map50:.4f}")
print(f"  VAL  mAP@0.5:0.95   : {val_metrics.box.map:.4f}")
print(f"  VAL  Precision      : {val_metrics.box.mp:.4f}")
print(f"  VAL  Recall         : {val_metrics.box.mr:.4f}")
print(f"{'═'*45}")

if len(class_names_model) > 1:
    print("\nPer-class mAP@0.5:")
    for i, name in enumerate(class_names_model):
        print(f"  {name:20s}: {val_metrics.box.maps[i]:.4f}")

print("\n✅ TRAINING & EVALUATION COMPLETE")
print(f"   best.pt location : {best_pt}")
