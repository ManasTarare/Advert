# ============================================================
# BILLBOARD DETECTOR — INFERENCE / TESTING ONLY
# No Google Drive needed. Upload your model manually.
# Run cell-by-cell in Google Colab.
# ============================================================

# ── CELL 1: Install dependencies ────────────────────────────
!pip install -q ultralytics supervision

import os, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from google.colab import files

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU — Go to Runtime → Change runtime type → T4 GPU")


# ── CELL 2: Upload your trained model (best.pt) ─────────────
print("\n📦 Please upload your trained model file (best.pt):")
uploaded_model = files.upload()

model_file = next((f for f in uploaded_model if f.endswith('.pt')), None)
if model_file is None:
    raise FileNotFoundError("❌ No .pt file found. Please upload best.pt")

MODEL_PATH = f"/content/{model_file}"
with open(MODEL_PATH, 'wb') as f:
    f.write(uploaded_model[model_file])

print(f"✅ Model saved to: {MODEL_PATH}")

from ultralytics import YOLO
model = YOLO(MODEL_PATH)
CLASS_NAMES = list(model.names.values())
print(f"✅ Model loaded  — Classes: {CLASS_NAMES}")


# ── CELL 3: Helper functions ─────────────────────────────────

def bytes_to_bgr(raw_bytes):
    return cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)


def auto_tune(image, yolo_model, base_conf=0.25):
    """Auto-select conf threshold and blend alpha based on image stats."""
    results = yolo_model(image, conf=base_conf, verbose=False)[0]
    boxes   = results.boxes

    final = []
    conf_thr = base_conf

    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.cpu().numpy()
        if len(confs) > 0:
            conf_thr = float(np.percentile(confs, 25))
            conf_thr = max(base_conf, min(conf_thr, 0.70))

        for b in boxes:
            lbl  = yolo_model.names[int(b.cls[0])]
            conf = float(b.conf[0])
            if lbl == "billboard" and conf >= conf_thr:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                final.append((x1, y1, x2, y2, conf, lbl))

    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    alpha = float(np.clip(0.80 + (np.std(gray) / 128) * 0.16, 0.80, 0.96))
    return round(conf_thr, 2), round(alpha, 2), final


def detect_quad_in_roi(image, x1, y1, x2, y2, debug=False):
    """Try to fit a perspective quad inside the bounding box ROI."""
    pad   = 10
    rx1   = max(0, x1 - pad)
    ry1   = max(0, y1 - pad)
    rx2   = min(image.shape[1], x2 + pad)
    ry2   = min(image.shape[0], y2 + pad)
    roi   = image[ry1:ry2, rx1:rx2]

    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return _bbox_corners(x1, y1, x2, y2)

    cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)
    roi_area = (rx2 - rx1) * (ry2 - ry1)

    for cnt in cnts[:5]:
        area = cv2.contourArea(cnt)
        if area < roi_area * 0.05:
            continue
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            corners[:, 0] += rx1
            corners[:, 1] += ry1
            corners = _order_corners(corners)
            return corners

    return _bbox_corners(x1, y1, x2, y2)


def _bbox_corners(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _order_corners(pts):
    rect  = np.zeros((4, 2), dtype=np.float32)
    s     = pts.sum(axis=1)
    diff  = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_ad_perspective(frame, ad_img, corners, alpha=0.88):
    """Warp ad image into the billboard corners using perspective transform."""
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w < 10 or h < 10:
        return frame

    ad_resized = cv2.resize(ad_img, (w, h))
    src_pts    = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_pts    = corners.astype(np.float32)

    M   = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warp = cv2.warpPerspective(ad_resized, M, (frame.shape[1], frame.shape[0]))

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(int), 255)
    mask_3 = cv2.merge([mask, mask, mask])

    out = frame.copy()
    out = np.where(mask_3 > 0,
                   (alpha * warp + (1 - alpha) * frame).astype(np.uint8),
                   frame)
    return out


def draw_quad_overlay(frame, corners, label, conf, used_quad):
    color = (0, 200, 80) if used_quad else (0, 140, 255)
    pts   = corners.astype(int).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, 2)
    mode  = "QUAD" if used_quad else "BBOX"
    text  = f"{label} {conf:.2f} [{mode}]"
    tx, ty = int(corners[0][0]), max(int(corners[0][1]) - 8, 12)
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return frame


def process_image(image, ad_img, yolo_model, debug=False):
    """Full pipeline: detect → quad → warp → overlay."""
    conf_thr, alpha, boxes = auto_tune(image, yolo_model)
    report = [
        f"Auto-settings → conf ≥ {conf_thr}  |  blend α = {alpha:.2f}  |  {len(boxes)} billboard(s)"
    ]

    replaced  = image.copy()
    n_quad    = 0
    n_bbox_fb = 0

    for (x1, y1, x2, y2, conf_val, label) in boxes:
        corners  = detect_quad_in_roi(image, x1, y1, x2, y2, debug)
        bbox_c   = _bbox_corners(x1, y1, x2, y2)
        used_quad = not np.allclose(corners, bbox_c, atol=2.0)

        if used_quad:
            n_quad   += 1
            mode_str  = "QUAD (perspective)"
        else:
            n_bbox_fb += 1
            mode_str  = "BBOX fallback"

        area = (x2 - x1) * (y2 - y1)
        report.append(f"  • {label} ({conf_val:.2f})  {mode_str}  area={area}px²")

        if ad_img is not None:
            replaced = warp_ad_perspective(replaced, ad_img, corners, alpha)
        replaced = draw_quad_overlay(replaced, corners, label, conf_val, used_quad)

    return replaced, report, n_quad, n_bbox_fb


def show_comparison(orig, result, title="", report_lines=None):
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    axes[0].imshow(cv2.cvtColor(orig,   cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Detected / Ad Replaced", fontsize=13, fontweight="bold", color="#1a5c3a")
    axes[1].axis("off")
    if title:
        fig.suptitle(title, fontsize=10, color="#444")
    plt.tight_layout()
    plt.show()
    if report_lines:
        for line in report_lines:
            print("  " + line)


# ── CELL 4: State holder ─────────────────────────────────────
state = {"ad_img": None, "target_imgs": {}}


# ── CELL 5: STEP 1 — Upload Ad Image (optional) ──────────────
display(HTML("""
<div style='background:#1e3a5f;color:white;padding:12px 18px;border-radius:8px;
            font-family:monospace;font-size:14px;margin-bottom:6px'>
  <b>STEP 1 (optional):</b> Upload your Advertisement / Logo image (jpg, png)<br>
  <small style='opacity:.8'>Skip this if you only want to see detections without replacement.</small>
</div>"""))

btn_ad = widgets.Button(description="📤 Upload Ad Image", button_style="primary",
                        layout=widgets.Layout(width="220px", height="38px"))
out_ad = widgets.Output()

def on_ad(_):
    with out_ad:
        clear_output()
        up = files.upload()
        if not up:
            print("⚠️  Nothing selected.")
            return
        fname, raw = next(iter(up.items()))
        img = bytes_to_bgr(raw)
        if img is None:
            print(f"❌ Cannot decode {fname}")
            return
        state["ad_img"] = img
        print(f"✅ Ad loaded: {fname}  ({img.shape[1]}×{img.shape[0]})")
        plt.figure(figsize=(5, 3))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Ad Image")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

btn_ad.on_click(on_ad)
display(widgets.VBox([btn_ad, out_ad]))


# ── CELL 6: STEP 2 — Upload Target Images ────────────────────
display(HTML("""
<div style='background:#2d5a1b;color:white;padding:12px 18px;border-radius:8px;
            font-family:monospace;font-size:14px;margin:12px 0 6px'>
  <b>STEP 2:</b> Upload Target Image(s) containing billboards
</div>"""))

btn_tgt = widgets.Button(description="📤 Upload Target Image(s)", button_style="success",
                         layout=widgets.Layout(width="260px", height="38px"))
out_tgt = widgets.Output()

def on_tgt(_):
    with out_tgt:
        clear_output()
        up = files.upload()
        if not up:
            print("⚠️  Nothing selected.")
            return
        state["target_imgs"] = {}
        for fname, raw in up.items():
            img = bytes_to_bgr(raw)
            if img is not None:
                state["target_imgs"][fname] = img
                print(f"  ✅ {fname}  ({img.shape[1]}×{img.shape[0]})")
        print(f"\n📂 {len(state['target_imgs'])} image(s) ready.")

btn_tgt.on_click(on_tgt)
display(widgets.VBox([btn_tgt, out_tgt]))


# ── CELL 7: STEP 3 — Run inference ───────────────────────────
display(HTML("""
<div style='background:#4a1f6e;color:white;padding:12px 18px;border-radius:8px;
            font-family:monospace;font-size:14px;margin:12px 0 6px'>
  <b>STEP 3:</b> Run billboard detection &amp; ad replacement
  <br><small style='opacity:.8'>
    🟢 Green outline = perspective quad  |  🟠 Orange = bbox fallback
  </small>
</div>"""))

chk_save = widgets.Checkbox(value=True, description="Save output images (downloadable)", indent=False)
chk_debug = widgets.Checkbox(value=False, description="Debug mode (verbose)", indent=False)
display(chk_save)
display(chk_debug)

btn_run = widgets.Button(description="🚀 Run Detection", button_style="danger",
                         layout=widgets.Layout(width="220px", height="42px", margin="10px 0 0 0"))
out_run = widgets.Output()

def on_run(_):
    with out_run:
        clear_output()
        if not state["target_imgs"]:
            print("❌ Upload at least one target image first (Step 2).")
            return

        if state["ad_img"] is None:
            print("ℹ️  No ad image uploaded — showing detections only (no replacement).")

        os.makedirs("/content/billboard_outputs", exist_ok=True)

        for fname, orig in state["target_imgs"].items():
            print(f"━━ {fname}")
            replaced, report, nq, nb = process_image(
                orig, state["ad_img"], model, debug=chk_debug.value
            )
            title = f"{fname}  |  {nq} quad-detected  +  {nb} bbox-fallback"
            show_comparison(orig, replaced, title=title, report_lines=report)

            if chk_save.value:
                from pathlib import Path
                out_path = f"/content/billboard_outputs/result_{Path(fname).stem}.jpg"
                cv2.imwrite(out_path, replaced)
                print(f"  💾 Saved → {out_path}")
            print()

        print("🎉 Done!")
        print("\nLegend:")
        print("  🟢 Green  = perspective quad (ad follows actual billboard plane)")
        print("  🟠 Orange = bbox fallback (no clean quad found)")

        if chk_save.value:
            print(f"\n📁 Outputs saved to /content/billboard_outputs/")

btn_run.on_click(on_run)
display(widgets.VBox([btn_run, out_run]))


# ── CELL 8: Download results ──────────────────────────────────
# Run this cell AFTER Step 3 to download all output images.

import glob, zipfile
from pathlib import Path

output_files = glob.glob("/content/billboard_outputs/*.jpg")
if not output_files:
    print("⚠️  No output files found yet. Run Step 3 first.")
else:
    zip_path = "/content/billboard_results.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for fp in output_files:
            zf.write(fp, Path(fp).name)
    print(f"✅ Zipped {len(output_files)} file(s) → {zip_path}")
    files.download(zip_path)
    print("⬇️  Download started!")


# ── CELL 9: (Optional) Quick detection-only on a single image ─
# Use this for a fast test without the widget UI.
# Upload an image file and set its name below.

QUICK_TEST_IMAGE = "your_image.jpg"   # ← change this to your filename

if os.path.exists(f"/content/{QUICK_TEST_IMAGE}"):
    img = cv2.imread(f"/content/{QUICK_TEST_IMAGE}")
    results = model(img, conf=0.25, verbose=True)
    annotated = results[0].plot()

    plt.figure(figsize=(12, 7))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title(f"Quick Detection — {QUICK_TEST_IMAGE}", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("\nDetections:")
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls[0])]
        conf     = float(box.conf[0])
        xyxy     = [round(x, 1) for x in box.xyxy[0].tolist()]
        print(f"  {cls_name:20s}  conf={conf:.3f}  box={xyxy}")
else:
    print(f"⚠️  File not found: /content/{QUICK_TEST_IMAGE}")
    print("   Upload the image to /content/ or change QUICK_TEST_IMAGE above.")
