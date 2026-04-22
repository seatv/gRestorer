# gRestorer
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)

gRestorer is a **GPU-first** video pipeline for mosaic detection, restoration, and face swapping:

**NVDEC (PyNvVideoCodec) → RGB/RGBP → BGR → Detect → (Track → Clip Restore / Face Swap → Composite) → NVENC → FFmpeg remux**

It’s built to be **measurable first, optimized second**: the CLI prints per-stage timings so you can tune performance and quality surgically.

---

## What it does

1. **Decode** frames on GPU using `PyNvVideoCodec` (NVDEC)
2. Convert decoder output **RGB/RGBP → BGR** (LADA-style and face-swap models expect BGR ordering)
3. (Optional) **Detect mosaics** using a YOLO segmentation model (Ultralytics)
4. **Restore / transform**
   - `none`: passthrough baseline (decode + conversion + encode cost)
   - `pseudo`: draw ROI boxes + fill mosaic regions (visual sanity-check)
   - `pseudo_clip`: clip-mode pipeline (tracker/compositor validation)
   - `basicvsrpp`: clip restoration using a BasicVSR++-style model checkpoint
   - `face_swap`: ROI-authoritative face-swap pipeline with backend-specific workers
5. **Composite** restored patches back into the full frame using a **LADA-style blend mask** (mosaic path)
6. **Encode** on GPU using `PyNvVideoCodec` (NVENC)
7. **Remux** to MP4 with FFmpeg (optionally copying audio/subtitles from the source)

---

## CLI entry points

After `pip install -e .` you can use either style:

- `grestorer ...` — main pipeline (decode → [detect] → restore / swap → encode → remux)
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
  detector/    # mosaic detector wrappers (YOLO seg), face detector wrapper
  restorer/    # restorers: none, pseudo, pseudo_clip, basicvsrpp, face_swap
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
  --input  "D:\Videos\Test\sample.mp4" `
  --output "D:\Videos\Test\out_none.mp4" `
  --restorer none
```

### Pseudo mode (visualize detection)

```powershell
grestorer `
  --input  "D:\Videos\Test\sample.mp4" `
  --output "D:\Videos\Test\out_pseudo.mp4" `
  --restorer pseudo `
  --det-model "D:\Models\lada\lada_mosaic_detection_model_v4.pt" `
  --debug
```

### Clip-mode pseudo (validates tracker + compositor + drain)

```powershell
grestorer `
  --input  "D:\Videos\Test\sample.mp4" `
  --output "D:\Videos\Test\out_pseudo_clip.mp4" `
  --restorer pseudo_clip `
  --det-model "D:\Models\lada\lada_mosaic_detection_model_v4.pt" `
  --debug
```

### BasicVSR++ clip restoration

```powershell
grestorer `
  --input  "D:\Videos\Test\sample.mp4" `
  --output "D:\Videos\Test\out_basicvsrpp.mp4" `
  --restorer basicvsrpp `
  --det-model  "D:\Models\lada\lada_mosaic_detection_model_v4.pt" `
  --rest-model "D:\Models\lada\lada_mosaic_restoration_model_generic_v1.2.pth" `
  --debug
```

### Face swap

Face swap uses the same clip-oriented pipeline shell, but the ROI content is transformed by a face-swap backend instead of a mosaic restoration model.

Minimal example:

```powershell
grestorer `
  --input       "D:\Videos\Test\sample.mp4" `
  --output      "D:\Videos\Test\sample_swap.mp4" `
  --restorer    face_swap `
  --source-face "D:\Faces\source.jpg" `
  --swap-model  "D:\Models\faceswap\inswapper_128.onnx"
```

Typical HyperSwap / SimSwap runs are usually driven from `config.json`, because there are more tuning knobs than a one-liner is comfortable with.

---

## Face swapping (current architecture)

### High-level path

Current face swapping is **ROI-authoritative**:

1. Detect face ROIs in the clip
2. Stabilize the per-clip target face track
3. Optionally refine target landmarks
4. Run a backend-specific swap worker on the crop
5. Optionally run face enhancement
6. Optionally run occlusion preservation
7. Return the modified crop to the clip/frame pipeline

### Current validated backend paths

#### InSwapper
- Uses the legacy/native InsightFace paste-back path
- Best current “plug it in and it works” quality
- Good alignment and usually the least tuning pain

#### HyperSwap
- Uses a **native worker path**
- Follows a FaceFusion-style integration
- Uses whole-face aligned crop inference and native mask/paste-back
- This is the preferred path for HyperSwap; the shared compositor path was not a good fit

#### SimSwap
- Also uses a **native worker path**
- Follows a FaceFusion-style integration
- Properly scales and pastes back, but usually benefits more from enhancement than InSwapper or HyperSwap

### Important design note

The framework still contains a shared face compositor path for backends that return aligned backend results, but the currently validated face-swap backends above are all driven through **native `swap()` workers**.

That is intentional.

For the currently validated backends:
- **InSwapper**: native path
- **HyperSwap**: native path
- **SimSwap**: native path

So the `face_comp_*` knobs remain part of the framework surface, but they are **not the primary quality controls** for the native face-swap paths above.

---

## Face-swap tuning knobs

The face-swap pipeline exposes the following practical controls.

### Backend selection / model inputs
- `source_face_path` — source identity image
- `swap_model_path` — ONNX swap model
- `swap_backend` — explicit backend selection when you do not want backend autodetect
- `swap_input_size` — aligned crop size requested by the backend wrapper
- `provider` / `swap_provider` — `cpu`, `cuda`, `xpu`, or `auto`

### Landmark refinement / target stability
- `landmark_refiner_enabled`
- `landmark_model`
- `landmark_model_path`
- `landmark_provider`
- `landmark_score`

Practical note:
- Landmark refinement is worth enabling when you want better mask geometry and more stable face coverage on difficult clips.

### Face enhancement
- `face_enhancer_enabled`
- `face_enhancer_model_path`
- `face_enhancer_provider`
- `face_enhancer_blend`

Practical note:
- **SimSwap** often benefits noticeably from the enhancer.
- Start with `face_enhancer_blend` around `70-80` and adjust from there.

### Occlusion preservation
- `face_occluder_enabled`
- `face_occluder_model_path`
- `face_occluder_provider`
- `face_occluder_threshold`
- `face_occluder_blur`
- `face_occluder_blend`
- `face_occluder_invert`

Practical note:
- Useful when the target face is partially blocked by a microphone, hand, hair strand, or similar object.
- Start conservative: threshold near `0.5`, blur near `5`.

### Debug / validation
- `debug_enabled`
- `debug_dir`
- `debug_start`
- `debug_end`
- `material_change_mad_threshold`

Practical note:
- Use debug frame dumps to inspect crops, target selection, and post-swap changes without committing to a full long encode.

### Framework compositor knobs
These still exist in the shared face pipeline surface:
- `face_comp_mask_mode`
- `face_comp_geom_expand`
- `face_comp_mask_erode`
- `face_comp_mask_dilate`
- `face_comp_mask_blur`
- `face_comp_blend_mode`
- `face_comp_color_transfer`
- `face_comp_face_scale`
- `face_comp_debug`

Practical note:
- These are still useful for backends that genuinely use the shared compositor path.
- For the current native InSwapper / HyperSwap / SimSwap flows, they are **not** the primary tuning levers.

---

## Recommended face-swap starting points

### InSwapper
- Start here if you want the most reliable baseline
- Usually runs fine without enhancer
- Good default for “is the rest of the face-swap pipeline healthy?” testing

### HyperSwap
- Use the native HyperSwap worker path
- Start without enhancer and without occluder
- Turn on landmark refinement if edge coverage or crop geometry needs help

### SimSwap
- Use the native SimSwap worker path
- Expect to use enhancer more often than with InSwapper / HyperSwap
- Start with enhancer enabled and moderate blend

### What not to do
- Do not try to force HyperSwap through the shared compositor path
- Do not assume compositor face-scale/mask shaping is the main quality fix for native paths
- Treat experimental pixel-boost work as separate from the stable native backend paths unless explicitly validated

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
grestorer --input Mosaic.ts --output Restored.mp4 --det-model D:\Models\lada\lada_vr_mosaic_detection_model_1.0.pt --sbs --sbs-layout lr
```

### Troubleshooting SBS errors

**High-res decode limits (NVDEC):** Some SBS sources are **4320×2160** (5K). On some NVIDIA NVDEC generations, the max supported decode dimension is **4096 px** per side. In that case PyNvVideoCodec will fail with an error like:
- `Error code : 801` / `Resolution not supported on this GPU`

 gRestorer will fall back to **CPU decode** so the run can continue, but throughput will be slower.

### High-resolution inputs and NVDEC limits (automatic CPU decode fallback)

Some videos exceed the NVDEC decode limits of certain GPUs (common threshold: **4096 px** max in width/height). Example: **4320×2160**.

Symptoms:
- PyNvVideoCodec error like `Error code : 801` / `Resolution not supported on this GPU`
- Decode fails immediately (often on the first batch)

Behavior:
- gRestorer will automatically fall back to **CPU decode** to keep the pipeline running.
- Expect lower throughput due to CPU decode + GPU upload.

Workarounds:
- Use a GPU/NVDEC generation that supports the input resolution, or
- Downscale to ≤4096 width, or
- Pre-remux/convert problematic containers before processing.

### MP4 + HEVC playback compatibility (hvc1 vs hev1)

Some players (including certain Quest playback stacks) are picky about the MP4 video sample entry for HEVC:
- `hev1` can cause **jerky / broken** playback in some players
- `hvc1` is often the more compatible tag for MP4+HEVC

gRestorer now remuxes MP4 outputs with the correct tag:
- MP4 + HEVC ⇒ `-tag:v hvc1`
- MP4 + H.264 ⇒ `-tag:v avc1`

Quick check:
```powershell
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,codec_tag_string,width,height -of default=nw=1 "VIDEO.mp4"
```

How to fix an older file:
```PowerShell
ffmpeg -hide_banner -y -i "in.mp4" -c copy -tag:v hvc1 "out.mp4"
```

### TS/MPEG-TS inputs (recommended remux before processing)

Transport streams can carry odd timestamp behavior and are more likely to trip up tooling. If you hit weird decode behavior, remux first (no re-encode):
```PowerShell
ffmpeg -hide_banner -y -fflags +genpts -i "in.ts" -map 0 -c copy "in.mp4"
```

---

## Synth mosaic generator

Generate controlled SFW mosaics (fixed ROIs) for testing:

```powershell
grestorer-add-mosaic `
  --input  "D:\Videos\Test\sample.mp4" `
  --output "D:\Videos\Mosaic\sample-M3.mp4"
```

ROIs can be specified either via CLI (`--roi t,l,b,r`, repeatable) or in `config.json` under `synth_mosaic.rois`.

---

## Configuration (`config.json`)

`config.json` is optional; CLI flags override config values.

### Processing family selection

Unless explicitly specified, gRestorer operates on **mosaic restoration**.

Use:

- `"process": "mosaic"` for mosaic detection/restoration
- `"process": "face"` for face detection / face-swap workflows

If `process` is omitted, the CLI defaults to **mosaic** blocks:
- `mosaic_detection` then legacy `detection`
- `mosaic_restoration` then legacy `restoration`

Face workflows must set:

```json
{
  "process": "face"
}
```

Use `debug_enabled` as the runtime master switch. The `debug` object is only for debug settings/output locations.

Common mosaic knobs:

- `detection.batch_size`, `detection.imgsz`, `detection.conf_threshold`, `detection.iou_threshold`, `detection.fp16`
- `restoration.max_clip_length`, `restoration.clip_size`, `restoration.border_ratio`, `restoration.pad_mode`, `restoration.fp16`
- `roi_dilate` — expand ROI boxes (pixels) before cropping/restoring
- `encoder.*` — codec/preset/profile/qp and remux behavior

Common face knobs:

```json
{
  "process": "face",
  "restorer": "face_swap",
  "source_face_path": "D:\Faces\source.jpg",
  "swap_model_path": "D:\Models\faceswap\inswapper_128.onnx",
  "swap_backend": "auto",
  "swap_input_size": 256,
  "swap_provider": "cuda",

  "landmark_refiner_enabled": false,
  "landmark_refiner_model": "2dfan4",
  "landmark_refiner_model_path": "",
  "landmark_refiner_provider": "auto",
  "landmark_refiner_score": 0.5,

  "face_enhancer_enabled": false,
  "face_enhancer_model_path": "",
  "face_enhancer_provider": "auto",
  "face_enhancer_blend": 80,

  "face_occluder_enabled": false,
  "face_occluder_model_path": "",
  "face_occluder_provider": "auto",
  "face_occluder_threshold": 0.5,
  "face_occluder_blur": 5,
  "face_occluder_blend": 100,
  "face_occluder_invert": false,

  "fs_debug_enabled": false,
  "fs_debug_dir": "fs_debug",
  "fs_debug_start": -1,
  "fs_debug_end": -1,
  "fs_material_mad": 1.0
}
```

### Compositing / seams
Paste-back uses a **blend mask** to reduce visible ROI boundaries on the mosaic path. If you still see seams:
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

### Face-swap quality debugging
- First validate backend + model + source face with **InSwapper**
- Then test HyperSwap / SimSwap with enhancer and occluder off
- Use debug frame dumps to inspect target face selection and swap output before adding post-processing
- If SimSwap looks structurally right but soft, enable enhancer before changing anything else

---

## Acknowledgements

This project draws heavily from:

- **lada** – for the detection and restoration models and the original pipeline.
- **BasicVSR++** – for the underlying video restoration architecture.
- **InsightFace / InSwapper** – for the InSwapper face-swap path.
- **FaceFusion** – for practical integration patterns for HyperSwap / SimSwap native worker flows.

Please check the upstream projects for full training code, original implementations, and model weights.

---

## License

AGPL-3.0
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
