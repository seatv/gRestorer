# gRestorer
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)

gRestorer is a **GPU-first** video pipeline for mosaic detection and restoration:

**NVDEC (PyNvVideoCodec) → RGB/RGBP → BGR → Detect → (Track → Clip Restore → Composite) → NVENC → FFmpeg remux**

It’s built to be **measurable first, optimized second**: the CLI prints per-stage timings so you can tune performance and quality surgically.

---

## What it does

1. **Decode** frames on GPU using `PyNvVideoCodec` (NVDEC)
2. Convert decoder output **RGB/RGBP → BGR** (LADA-style models expect BGR ordering)
3. (Optional) **Detect mosaics** using a YOLO segmentation model (Ultralytics)
4. **Restore**
   - `none`: passthrough baseline (decode + conversion + encode cost)
   - `pseudo`: draw ROI boxes + fill mosaic regions (visual sanity-check)
   - `pseudo_clip`: clip-mode pipeline (tracker/compositor validation)
   - `basicvsrpp`: clip restoration using a BasicVSR++-style model checkpoint
5. **Composite** restored patches back into the full frame using a **LADA-style blend mask** (reduces visible paste-back seams)
6. **Encode** on GPU using `PyNvVideoCodec` (NVENC)
7. **Remux** to MP4 with FFmpeg (optionally copying audio/subtitles from the source)

---

## CLI entry points

After `pip install -e .` you can use either style:

- `grestorer ...` — main pipeline (decode → [detect] → restore → encode → remux)
- `grestorer-add-mosaic ...` — synth mosaic generator (for creating SFW test clips)

Module form also works:

- `python -m gRestorer.cli ...`
- `python -m gRestorer.cli.add_mosaic ...`

Both commands default to loading `./config.json` if present.

---

## Project layout

```text
gRestorer/
  cli/         # CLI entry + pipeline orchestration
  core/        # scene/clip tracking logic
  detector/    # mosaic detector wrapper (YOLO seg)
  restorer/    # restorers: none, pseudo, pseudo_clip, basicvsrpp
  utils/       # config + visualization helpers
  video/       # NVDEC/NVENC wrappers: decoder.py, encoder.py
  synthmosaic/ # mosaic addition functions
pyproject.toml
requirements.txt
requirements.torch-cu128.txt
config.json    # optional; loaded by CLI if present
README.md
```

---

## Install (Windows / PowerShell)

### Prereqs
- Python **3.11+** (3.13 recommended)
- **FFmpeg** available on PATH (needed for remuxing / `ffprobe` troubleshooting)
- For GPU decode/encode: NVIDIA driver + NVDEC/NVENC-capable GPU and `PyNvVideoCodec` working

### 1) Clone the repository

```powershell
git clone <REPO_URL>
cd gRestorer
```

### 2) Create and activate a venv

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### 3) Install PyTorch (IMPORTANT: do this first)

Ultralytics will pull **CPU-only torch** if torch isn’t installed yet. To avoid the YOLO seg mask failure seen with newer combos, this repo uses a **known-good pin**:

- `torch==2.9.1` + `torchvision==0.24.1`
- `ultralytics==8.3.243`

CUDA 12.8 wheels example:

```powershell
pip install -r requirements.torch-cu128.txt
```

Verify:

```powershell
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print('cuda ver', torch.version.cuda)"
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
```

For CPU-only or Intel XPU installs, install the appropriate torch build first (per the official PyTorch instructions), then continue below.

### 4) Install the rest of the dependencies

```powershell
pip install -r requirements.txt
```

### 5) Install gRestorer (editable, recommended)

```powershell
pip install -e .
python -m gRestorer.cli --help
```

---

## Usage

### Baseline passthrough (no detection, no restore)

```powershell
grestorer `
  --input  "D:\\Videos\\Test\\sample.mp4" `
  --output "D:\\Videos\\Test\\out_none.mp4" `
  --restorer none
```

### Pseudo mode (visualize detection)

```powershell
grestorer `
  --input  "D:\\Videos\\Test\\sample.mp4" `
  --output "D:\\Videos\\Test\\out_pseudo.mp4" `
  --restorer pseudo `
  --det-model "D:\\Models\\lada\\lada_mosaic_detection_model_v4.pt" `
  --debug
```

### Clip-mode pseudo (validates tracker + compositor + drain)

```powershell
grestorer `
  --input  "D:\\Videos\\Test\\sample.mp4" `
  --output "D:\\Videos\\Test\\out_pseudo_clip.mp4" `
  --restorer pseudo_clip `
  --det-model "D:\\Models\\lada\\lada_mosaic_detection_model_v4.pt" `
  --debug
```

### BasicVSR++ clip restoration

```powershell
grestorer `
  --input  "D:\\Videos\\Test\\sample.mp4" `
  --output "D:\\Videos\\Test\\out_basicvsrpp.mp4" `
  --restorer basicvsrpp `
  --det-model  "D:\\Models\\lada\\lada_mosaic_detection_model_v4.pt" `
  --rest-model "D:\\Models\\lada\\lada_mosaic_restoration_model_generic_v1.2.pth" `
  --debug
```
---

## SBS (Side-by-Side) 3D videos

gRestorer supports **SBS** (Left/Right) videos, including a seam-safe ROI mode so restored regions don’t create a visible “split seam” down the center line.

### Why SBS needs special handling
In SBS, the left and right views meet at a vertical boundary. If the detector produces an ROI that crosses that boundary, you can get:
- discontinuities at the center seam, or
- a restored patch that “belongs” to only one eye.

To avoid this, gRestorer can enforce **seam-safe ROIs**:
- ROIs are clipped or adjusted so they do not straddle the SBS seam.
- Optionally, detection can be performed per-eye (det-split) so each half is analyzed independently.

### Recommended settings for SBS
- Keep seam-safe ROI handling enabled for SBS content.
- Keep `restoration.feather_radius = 0` (paste-back already uses a blend mask).
- If you see occasional ROI jitter near the seam, increase `roi_dilate` slightly (+2 px).

### SBS example
```powershell
gRestorer --input Mosaic.ts --output Restored.mp4 --det-model D:\Models\lada\lada_vr_mosaic_detection_model_1.0.pt --sbs --sbs-layout lr
```

### Troubleshooting SBS errors
If an SBS file fails:
1) Confirm ffprobe can read it and that the frame rate is sane.
2) Try a short cut (1–2 seconds) to reproduce quickly:
```powershell
ffmpeg -ss 0 -to 2 -i "INPUT.mp4" -c copy "SBS_short.mp4"
```
3) Re-run gRestorer on the short clip with `--debug` and capture stderr output.

---

## Synth mosaic generator

Generate controlled SFW mosaics (fixed ROIs) for testing:

```powershell
grestorer-add-mosaic `
  --input  "D:\\Videos\\Test\\sample.mp4" `
  --output "D:\\Videos\\Mosaic\\sample-M3.mp4"
```

ROIs can be specified either via CLI (`--roi t,l,b,r`, repeatable) or in `config.json` under `synth_mosaic.rois`.

---

## Configuration (`config.json`)

`config.json` is optional; CLI flags override config values.

Common knobs:

- `detection.batch_size`, `detection.imgsz`, `detection.conf_threshold`, `detection.iou_threshold`, `detection.fp16`
- `restoration.max_clip_length`, `restoration.clip_size`, `restoration.border_ratio`, `restoration.pad_mode`, `restoration.fp16`
- `roi_dilate` — expand ROI boxes (pixels) before cropping/restoring
- `encoder.*` — codec/preset/profile/qp and remux behavior

### Compositing / seams
Paste-back uses a **blend mask** to reduce visible ROI boundaries. If you still see seams:
- keep `restoration.feather_radius` at `0` (recommended)
- optionally increase `roi_dilate` slightly (+2 px)

---

## Output timings

The pipeline reports:
- per-stage timings (`decode / det / track / restore / encode`)
- processing time without mux
- total time with mux (FFmpeg remux duration shown separately)

---

## Troubleshooting

### Verify frame counts (source of truth)

```powershell
ffprobe -v error -select_streams v:0 -count_frames `
  -show_entries stream=nb_read_frames -of default=nk=1:nw=1 "VIDEO.mp4"
```

### Ultralytics YOLO seg mask failure (KeyError in protos / process_mask)

If you see seg mask indexing errors with newer Ultralytics/Torch combos:
- Pin to `ultralytics==8.3.243`
- Install CUDA torch first (`torch==2.9.1`, `torchvision==0.24.1`) using `requirements.torch-cu128.txt`

### Detection misses small mosaics
- Increase `detection.imgsz` (e.g., 640 → 1280)
- For synth mosaics, use a sufficiently large mosaic block size so artifacts survive scaling

---

## Acknowledgements

This project draws heavily from:

- **lada** – for the detection and restoration models and the original pipeline.
- **BasicVSR++** – for the underlying video restoration architecture.

Please check the upstream projects for full training code, original implementations, and model weights.

---

## License

AGPL-3.0
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
