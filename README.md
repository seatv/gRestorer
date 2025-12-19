# gRestorer

gRestorer is an **NVDEC → (RGB/RGBP) → BGR → Detect → Restore → NVENC** video pipeline designed to be **measurable first, optimized second**.

Right now it focuses on two baseline modes:

- **none**: no-op restorer (baseline throughput + format conversion + encode cost)
- **pseudo**: uses the detector output to draw ROI boundaries and shade detected mosaic regions (visual sanity-check)

The long-term goal is a LADA-faithful restoration path (BasicVSR++-style) with minimal CPU↔GPU chatter.

## What it does today

1) **Decode** frames on GPU using `PyNvVideoCodec` (NVDEC)  
2) Convert decoder output **RGB/RGBP → BGR** (LADA expects BGR ordering)  
3) (Optional) **Detect mosaics** using a YOLO segmentation model (`ultralytics`)  
4) **Restore**
   - `none`: passthrough
   - `pseudo`: draw box + apply semi-transparent fill inside mosaic mask
5) **Pack** to the encoder-required 4-channel packed format
6) **Encode** on GPU using `PyNvVideoCodec` (NVENC)

The CLI prints **per-stage timing** so you can see where the time goes and optimize surgically.

## Project layout

```
gRestorer/
  cli/         # CLI entry + pipeline orchestration for gRestorer & mosaic
  core/        # scene/clip logic (next phase)
  detector/    # mosaic detector wrapper (YOLO)
  restorer/    # restorers: none, pseudo, (future) grestorer
  utils/       # config + visualization helpers
  video/       # NVDEC/NVENC wrappers: decoder.py, encoder.py
  synthmosaic/ # mosaic addition functions
pyproject.toml
requirements.txt
config.json   # optional; loaded by CLI if present
README.md
```

## Install

### 1) Create a venv (Windows / PowerShell)

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Install PyTorch

Install the torch build that matches your machine (CUDA / CPU / Intel XPU).
Example (CUDA builds are hosted by PyTorch):

```powershell
# Example only — choose the correct index URL for your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Run via module (recommended during development)

```powershell
python -m gRestorer.cli --input "D:\Videos\Test\sample.mp4" --output "D:\Videos\Test\out_none.mp4" --restorer none
```

### Run pseudo mode (requires detector model)

```powershell
python -m gRestorer.cli `
  --input  "D:\Videos\Test\sample.mp4" `
  --output "D:\Videos\Test\out_pseudo.mp4" `
  --restorer pseudo `
  --det-model "D:\Models\lada\lada_mosaic_detection_model_v3.1_accurate.pt"
```
# gRestorer

gRestorer is an NVDEC → (RGBP) → BGR → Detect → Restore → NVENC pipeline designed to be **measurable first** and then optimized until it matches (and hopefully beats) LADA throughput.

## CLI commands

- `gRestorer` — main pipeline (decode → [detect] → restore → encode)
- `mosaic` — synthetic mosaic generator (for creating SFW test clips)

## Install (Windows / PowerShell)

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .

### Install as an editable package (optional)

```powershell
pip install -e .
gRestorer --help
```

## Configuration

You can set defaults in `config.json` and override them on the command line.

Example `config.json`:

```json
{
  "gpu_id": 0,
  "batch_size": 8,
  "restorer": "pseudo",
  "det_imgsz": 640,
  "det_conf": 0.25,
  "det_iou": 0.45,
  "visualization": {
    "box_color": [0, 255, 0],
    "box_thickness": 2,
    "fill_color": [128, 128, 128],
    "fill_opacity": 0.5
  }
}
```

CLI overrides always win.

## Performance notes (current reality)

- **RGBP is usually the best decoder output** for ML pipelines because models want BCHW.
- gRestorer supports **RGB and RGBP** and converts to **BGR** for the LADA-style detector/restoration world.
- The current YOLO path may still incur **CPU work** in preprocessing (Ultralytics/letterbox). That’s a planned optimization.

## Roadmap

- [ ] GPU-only detector preprocessing (torch letterbox/pad/normalize on GPU)
- [ ] Scene/clip tracking (group detections into stable clips)
- [ ] LADA-faithful restoration (BasicVSR++-style) integrated into the clip pipeline
- [ ] Backends: CPU / CUDA / Intel XPU (decode/encode strategies per backend)
- [ ] Expose ffmpeg-style decode/encode options (when non-NVIDIA paths are added)

## License

AGPL-3.0 (see `pyproject.toml`).
