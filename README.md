# gRestorer
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)

gRestorer is a **GPU-first** video pipeline for mosaic detection and restoration:

**NVDEC (PyNvVideoCodec) → RGB/RGBP → BGR → Detect → (Track → Clip Restore → Composite) → NVENC → FFmpeg remux**

It’s built to be **measurable first, optimized second**: the CLI prints per-stage timings so you can tune performance and quality surgically.

## What it does

1. **Decode** frames on GPU using `PyNvVideoCodec` (NVDEC)
2. Convert decoder output **RGB/RGBP → BGR** (LADA-style models expect BGR ordering)
3. (Optional) **Detect mosaics** using a YOLO segmentation model (Ultralytics)
4. **Restore**
   - `none`: passthrough baseline (decode + conversion + encode cost)
   - `pseudo`: draw ROI boxes + fill mosaic regions (visual sanity-check)
   - `pseudo_clip`: clip-mode pipeline (tracker/compositor validation)
   - `basicvsrpp`: clip restoration using a BasicVSR++-style model checkpoint
5. **Encode** on GPU using `PyNvVideoCodec` (NVENC)
6. **Remux** to MP4 with FFmpeg (optionally copying audio/subtitles from the source)

## CLI commands

- `python -m gRestorer.cli` — main pipeline (decode → [detect] → restore → encode → remux)
- `python -m gRestorer.cli.add-mosaic` — GPU synth mosaic generator (for creating SFW test clips)

Both commands default to loading `./config.json` if present.

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
config.json    # optional; loaded by CLI if present
README.md
```

## Install (Windows / PowerShell)

### 1) Create and activate a venv

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Install PyTorch

Install the torch build that matches your machine (CUDA / CPU / Intel XPU). Example (CUDA):

```powershell
# Example only — choose the correct index URL for your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 4) Install gRestorer (editable, recommended for dev)

```powershell
pip install -e .
python -m gRestorer.cli --help
```

## Usage

### Baseline passthrough

```powershell
python -m gRestorer.cli `
  --input  "D:\Videos\Test\sample.mp4" `
  --output "D:\Videos\Test\out_none.mp4" `
  --restorer none
```

### Pseudo mode (visualize detection)

```powershell
python -m gRestorer.cli `
  --input  "D:\Videos\Test\sample.mp4" `
  --output "D:\Videos\Test\out_pseudo.mp4" `
  --restorer pseudo `
  --det-model "D:\Models\lada\lada_mosaic_detection_model_v3.1_accurate.pt" `
  --debug
```

### Clip-mode pseudo (validates tracker + compositor + drain)

```powershell
python -m gRestorer.cli `
  --input  "D:\Videos\Test\sample.mp4" `
  --output "D:\Videos\Test\out_pseudoClip.mp4" `
  --restorer pseudo_clip `
  --det-model "D:\Models\lada\lada_mosaic_detection_model_v3.1_accurate.pt" `
  --debug
```

### BasicVSR++ clip restoration

```powershell
python -m gRestorer.cli `
  --input  "D:\Videos\Test\sample.mp4" `
  --output "D:\Videos\Test\out_basicvsrpp.mp4" `
  --restorer basicvsrpp `
  --det-model  "D:\Models\lada\lada_mosaic_detection_model_v3.1_accurate.pt" `
  --rest-model "D:\Models\lada\lada_mosaic_restoration_model_generic_v1.2.pth" `
  --debug
```

## Synth mosaic generator

Generate controlled SFW mosaics (fixed ROIs) for testing:

```powershell
python -m gRestorer.cli.add-mosaic `
  --input  "D:\Videos\Test\sample.mp4" `
  --output "D:\Videos\Mosaic\sample-M3.mp4"
```

ROIs can be specified either via CLI (`--roi t,l,b,r`, repeatable) or in `config.json` under `synth_mosaic.rois`.

## Configuration

`config.json` is optional; CLI flags override config values.

Common knobs:

- `detection.batch_size`, `detection.imgsz`, `detection.conf_threshold`, `detection.iou_threshold`, `detection.fp16`
- `restoration.max_clip_length`, `restoration.clip_size`, `restoration.border_ratio`, `restoration.pad_mode`, `restoration.fp16`
- `restoration.feather_radius` — compositor feather blending at ROI boundary  
  - **Recommended default: `0`** (larger values can reintroduce mosaic-edge artifacts)
- `roi_dilate` — expand ROI boxes (pixels) before cropping/restoring
- `encoder.*` — codec/preset/profile/qp and remux behavior
  - Remux uses FFmpeg and may optionally copy audio/subtitles from the input (if enabled in your encoder settings)

## Output timings

The pipeline reports:
- per-stage timings (`decode / det / track / restore / encode`)
- processing time without mux
- total time with mux (FFmpeg remux duration shown separately)

## Troubleshooting

### Verify frame counts (source of truth)

```powershell
ffprobe -v error -select_streams v:0 -count_frames `
  -show_entries stream=nb_read_frames -of default=nk=1:nw=1 "VIDEO.mp4"
```

### ROI boundary seam

- Set `restoration.feather_radius` to `0` (recommended)
- If needed: increase `roi_dilate` slightly (+2 px)

### Detection misses small mosaics

- Increase `detection.imgsz` (e.g., 640 → 1280)
- For synth mosaics, use a sufficiently large mosaic block size so artifacts survive scaling

## Acknowledgements

This project draws heavily from:

- **lada** – for the detection and restoration models and the original mmagic‑based pipeline.![](https://github.com/ladaapp/lada/blob/main/packaging/flatpak/share/io.github.ladaapp.lada.png)
- **BasicVSR++** – for the underlying video restoration architecture.

Please check the upstream projects for full training code, original implementations, and model weights.

## License

AGPL-3.0
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)

![lada](https://github.com/ladaapp/lada/blob/main/packaging/flatpak/share/io.github.ladaapp.lada.png)