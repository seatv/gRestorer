from pathlib import Path
import csv
import json

import cv2
import numpy as np


DEBUG_DIR = Path(r"D:\Results\mosaic_paste_debug")
OUT_CSV = DEBUG_DIR / "temporal_metrics.csv"


def load_gray(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img


def load_color(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return img


def align_common_hw(*arrays):
    """
    Crop all arrays to the minimum common HxW.
    Preserves channel count for color arrays.
    """
    heights = [arr.shape[0] for arr in arrays if arr is not None]
    widths = [arr.shape[1] for arr in arrays if arr is not None]
    h = min(heights)
    w = min(widths)

    aligned = []
    for arr in arrays:
        if arr is None:
            aligned.append(None)
        else:
            if arr.ndim == 2:
                aligned.append(arr[:h, :w])
            else:
                aligned.append(arr[:h, :w, ...])
    return aligned


def bbox_from_mask(mask_u8):
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def iou(a, b):
    a, b = align_common_hw(a, b)
    a_bin = a > 0
    b_bin = b > 0
    inter = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()
    return float(inter / union) if union else 1.0


def ring_from_alpha(alpha_u8):
    a = alpha_u8.astype(np.float32) / 255.0
    return ((a > 0.01) & (a < 0.99)).astype(np.uint8)


def centroid(mask_u8):
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return (None, None)
    return (float(xs.mean()), float(ys.mean()))


def mae(a, b, region=None):
    if region is None:
        a, b = align_common_hw(a, b)
        return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))

    a, b, region = align_common_hw(a, b, region)
    idx = region > 0
    if not np.any(idx):
        return 0.0
    return float(np.mean(np.abs(a.astype(np.float32)[idx] - b.astype(np.float32)[idx])))


def edge_energy(img_u8, region=None):
    if region is None:
        gray = img_u8
    else:
        gray, region = align_common_hw(img_u8, region)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    if region is None:
        return float(mag.mean())

    idx = region > 0
    if not np.any(idx):
        return 0.0
    return float(mag[idx].mean())


groups = {}
for p in DEBUG_DIR.glob("f*_clip*.json"):
    stem = p.stem
    groups[stem] = {
        "json": p,
        "alpha_actual": p.with_name(stem + "_alpha_actual.png"),
        "alpha_legacy": p.with_name(stem + "_alpha_legacy.png"),
        "mask_resized": p.with_name(stem + "_mask_resized.png"),
        "orig_roi": p.with_name(stem + "_orig_roi.png"),
        "restored_roi": p.with_name(stem + "_restored_roi.png"),
        "final_roi": p.with_name(stem + "_final_roi.png"),
    }


def sort_key(k):
    # f000001_clip0000
    fpart, cpart = k.split("_")
    return (int(fpart[1:]), int(cpart[4:]))


rows = []
prev = None

for key in sorted(groups.keys(), key=sort_key):
    files = groups[key]
    meta = json.loads(files["json"].read_text(encoding="utf-8"))

    alpha = load_gray(files["alpha_actual"])
    alpha_legacy = load_gray(files["alpha_legacy"])
    mask = load_gray(files["mask_resized"])
    orig = load_color(files["orig_roi"])
    restored = load_color(files["restored_roi"])
    final = load_color(files["final_roi"])

    # Align everything spatially for per-frame metrics
    alpha, alpha_legacy, mask, orig, restored, final = align_common_hw(
        alpha, alpha_legacy, mask, orig, restored, final
    )

    ring = ring_from_alpha(alpha)
    support = (alpha > 0).astype(np.uint8) * 255
    bbox = bbox_from_mask(support)
    cx, cy = centroid(support)

    row = {
        "key": key,
        "frame": meta.get("frame_idx"),
        "clip": meta.get("clip_idx"),
        "h": int(alpha.shape[0]),
        "w": int(alpha.shape[1]),
        "alpha_mean": float(alpha.mean() / 255.0),
        "alpha_soft_pct": float(np.mean((alpha > 0) & (alpha < 255))),
        "support_area_pct": float(np.mean(alpha > 0)),
        "ring_area_pct": float(np.mean(ring > 0)),
        "final_vs_rest_mae_all": mae(final, restored),
        "final_vs_rest_mae_ring": mae(final, restored, ring),
        "final_vs_orig_mae_ring": mae(final, orig, ring),
        "rest_edge_ring": edge_energy(cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY), ring),
        "final_edge_ring": edge_energy(cv2.cvtColor(final, cv2.COLOR_BGR2GRAY), ring),
        "legacy_alpha_mean": float(alpha_legacy.mean() / 255.0),
        "bbox_x1": bbox[0] if bbox else None,
        "bbox_y1": bbox[1] if bbox else None,
        "bbox_x2": bbox[2] if bbox else None,
        "bbox_y2": bbox[3] if bbox else None,
        "centroid_x": cx,
        "centroid_y": cy,
    }

    if prev is None or prev["clip"] != row["clip"]:
        row.update({
            "alpha_iou_prev": None,
            "support_iou_prev": None,
            "ring_iou_prev": None,
            "centroid_shift_prev": None,
            "bbox_l1_prev": None,
        })
    else:
        prev_alpha = load_gray(groups[prev["key"]]["alpha_actual"])
        prev_alpha, alpha_for_iou = align_common_hw(prev_alpha, alpha)
        prev_ring = ring_from_alpha(prev_alpha)
        prev_support = (prev_alpha > 0).astype(np.uint8) * 255

        prev_bbox = (
            prev["bbox_x1"],
            prev["bbox_y1"],
            prev["bbox_x2"],
            prev["bbox_y2"],
        )

        bbox_l1 = None
        if bbox and prev_bbox[0] is not None:
            bbox_l1 = int(
                abs(bbox[0] - prev_bbox[0]) +
                abs(bbox[1] - prev_bbox[1]) +
                abs(bbox[2] - prev_bbox[2]) +
                abs(bbox[3] - prev_bbox[3])
            )

        cshift = None
        if cx is not None and prev["centroid_x"] is not None:
            cshift = float(((cx - prev["centroid_x"]) ** 2 + (cy - prev["centroid_y"]) ** 2) ** 0.5)

        row.update({
            "alpha_iou_prev": iou(alpha, prev_alpha),
            "support_iou_prev": iou(support, prev_support),
            "ring_iou_prev": iou(ring * 255, prev_ring * 255),
            "centroid_shift_prev": cshift,
            "bbox_l1_prev": bbox_l1,
        })

    rows.append(row)
    prev = row


if rows:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {OUT_CSV}")
else:
    print(f"No debug JSON files found in: {DEBUG_DIR}")