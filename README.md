# 🪧 Billboard Detector & Ad Replacer

A YOLOv8-powered pipeline that **detects billboards** in images and **replaces them with custom advertisements** using perspective-correct warping. Built for Google Colab with a clean widget-based UI.

---

## ✨ Features

- **YOLOv8 object detection** with auto-selected model size (S / M / L) based on dataset scale
- **Perspective quad fitting** — detects the actual billboard plane using contour analysis, not just a bounding box
- **Auto-tuned inference** — confidence threshold and blend alpha are dynamically adjusted per image
- **Ad replacement** — warps your custom ad image onto detected billboards with realistic perspective
- **Visual fallback** — falls back to bounding-box overlay when no clean quad is found
- **Colab widget UI** — step-by-step upload buttons, checkboxes, and one-click download of results
- **Training curves & metrics** — mAP@0.5, mAP@0.5:0.95, precision, recall plotted automatically

---

## 📁 Project Structure

```
├── re_adv.py        # Part 1 & 2: Dataset preparation + YOLOv8 training (Colab)
├── testing.py       # Part 3: Inference, ad replacement & result download (Colab)
```

---

## 🚀 Quickstart

### Prerequisites

- Google Colab with a **T4 GPU** runtime  
  *(Runtime → Change runtime type → T4 GPU)*

### Training (`re_adv.py`)

| Step | Cell | What it does |
|------|------|-------------|
| 1 | Install | Installs `ultralytics` + `supervision` |
| 2 | Upload | Upload your dataset `.zip` |
| 3–4 | Extract | Auto-locates `images/` and `annotations/` folders |
| 5–6 | Inspect | Reads CSVs, detects coordinate format (pixel vs. normalized) |
| 7 | Match | Pairs images with annotation CSVs |
| 8 | Convert | Converts to YOLO format with 80/10 train/val split |
| 9 | YAML | Writes `data.yaml` |
| 10 | Visualize | Shows ground-truth boxes on sample images |
| Train | Auto | Picks YOLOv8s/m/l based on dataset size, trains for 200 epochs |
| Eval | Metrics | Plots loss curves, mAP, precision, recall on val split |

**Dataset format expected:**

```
dataset.zip
├── images/
│   ├── img001.jpg
│   └── ...
└── annotations/   (or labels/ or csvs/)
    ├── img001.csv
    └── ...
```

Each CSV must have columns: `label`, `x1`, `y1`, `x2`, `y2`  
(pixel or normalized coordinates are both supported — auto-detected)

---

### Inference (`testing.py`)

Run cell-by-cell in a **separate Colab notebook** (no Drive needed):

```
Cell 1 → Install dependencies
Cell 2 → Upload your trained best.pt
Cell 3 → Helper functions loaded
Cell 4 → State initialized
Cell 5 → (Optional) Upload your ad/logo image
Cell 6 → Upload target image(s) with billboards
Cell 7 → Run detection + ad replacement
Cell 8 → Download all results as a ZIP
Cell 9 → Quick single-image test (no widgets)
```

**Output legend:**

| Outline color | Meaning |
|---------------|---------|
| 🟢 Green | Perspective quad — ad follows the actual billboard plane |
| 🟠 Orange | Bounding-box fallback — no clean quad detected |

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Image size | 640 |
| Optimizer | AdamW |
| LR schedule | Cosine annealing |
| Warmup epochs | 5 |
| Early stopping patience | 40 |
| Augmentations | Mosaic, MixUp, HSV jitter, affine, flip |
| Model (auto-selected) | YOLOv8s `(<500 imgs)` · YOLOv8m `(<2000)` · YOLOv8l `(2000+)` |

---

## 🧠 How Ad Replacement Works

```
Input image
    └── YOLOv8 detection → bounding boxes
            └── Contour analysis in each ROI
                    ├── 4-point quad found? → Perspective warp ad onto quad
                    └── No quad? → Warp ad onto bounding box corners
                            └── Alpha blend result onto original frame
```

The `auto_tune()` function picks the confidence threshold at the 25th percentile of detected confidences (clamped between `base_conf` and 0.70) and sets the blend alpha based on image contrast — so results adapt to each scene automatically.

---

## 📦 Dependencies

```
ultralytics
supervision
opencv-python
numpy
matplotlib
torch
PyYAML
pandas
ipywidgets
```

All installed automatically via `!pip install -q ultralytics supervision` in the first cell.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
