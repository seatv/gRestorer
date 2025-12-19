"""
copy_stream_gpu.py

Decode -> convert (RGBP -> packed) -> encode, all on GPU, using DLPack.

Your Decoder uses output_color_type=RGBP (planar CHW) and device memory.
Your Encoder is created with format="ABGR" and GPU input (usecpuinputbuffer=False).

Run:
  python copy_stream_gpu.py --input "D:\\Videos\\Test\\sample.mp4" --output "D:\\Videos\\Test\\out.hevc"
"""

from __future__ import annotations

import argparse
import sys
from typing import Literal, Optional

import torch

from decoder import Decoder
from encoder import Encoder


PackMode = Literal["abgr", "argb"]


def _ensure_uint8(rgb: torch.Tensor, bit_depth: int) -> torch.Tensor:
    """
    Ensure uint8 samples. If the stream is >8-bit, downshift to 8-bit on GPU.
    """
    if rgb.dtype == torch.uint8:
        return rgb

    if rgb.dtype in (torch.uint16, torch.int16, torch.int32, torch.int64):
        # Example: 10-bit typically stored in 16-bit container.
        shift = max(0, int(bit_depth) - 8)
        if shift > 0:
            return (rgb >> shift).to(torch.uint8)
        return rgb.to(torch.uint8)

    # Fallback: clamp + cast on GPU
    return rgb.clamp(0, 255).to(torch.uint8)


def rgbp_to_packed(
    rgbp_chw: torch.Tensor,
    out_hwc4: Optional[torch.Tensor],
    pack: PackMode,
) -> torch.Tensor:
    """
    Convert planar RGB (CHW, uint8) -> packed HWC4 (uint8) for NVENC.

    pack="abgr": writes bytes as RGBA (R,G,B,A). This matches ABGR word-order on little-endian.
    pack="argb": writes bytes as BGRA (B,G,R,A). This matches ARGB word-order on little-endian.

    NOTE: This function does one unavoidable GPU copy into a packed buffer.
    """
    if rgbp_chw.ndim != 3 or rgbp_chw.shape[0] != 3:
        raise ValueError(f"Expected RGBP tensor shaped [3,H,W], got {tuple(rgbp_chw.shape)}")

    # bgr is a *view* (no copy): channel swap only
    bgr_chw = rgbp_chw.flip(0)  # [3,H,W] with planes B,G,R

    h = int(bgr_chw.shape[1])
    w = int(bgr_chw.shape[2])

    if out_hwc4 is None or out_hwc4.shape != (h, w, 4) or out_hwc4.dtype != torch.uint8:
        out_hwc4 = torch.empty((h, w, 4), device=bgr_chw.device, dtype=torch.uint8)

    # Alpha channel = opaque
    out_hwc4[..., 3].fill_(255)

    b = bgr_chw[0]
    g = bgr_chw[1]
    r = bgr_chw[2]

    if pack == "abgr":
        # bytes: R,G,B,A
        out_hwc4[..., 0].copy_(r)
        out_hwc4[..., 1].copy_(g)
        out_hwc4[..., 2].copy_(b)
    elif pack == "argb":
        # bytes: B,G,R,A
        out_hwc4[..., 0].copy_(b)
        out_hwc4[..., 1].copy_(g)
        out_hwc4[..., 2].copy_(r)
    else:
        raise ValueError(f"Unknown pack mode: {pack}")

    return out_hwc4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input video path")
    ap.add_argument("--output", required=True, help="Output bitstream path (e.g., .hevc / .h264)")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device id")
    ap.add_argument("--batch", type=int, default=32, help="Decode batch size")
    ap.add_argument("--codec", default="hevc", choices=["hevc", "h264"], help="NVENC codec")
    ap.add_argument("--preset", default="p7", help="NVENC preset (p1..p7)")
    ap.add_argument("--profile", default="main", help="NVENC profile (e.g., main, main10)")
    ap.add_argument("--qp", type=int, default=23, help="NVENC QP (0 = lossless-ish)")
    ap.add_argument(
        "--pack",
        default="abgr",
        choices=["abgr", "argb"],
        help="How to pack bytes into HWC4 before Encode(). Try the other if colors look swapped.",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False. This pipeline requires CUDA.", file=sys.stderr)
        return 2

    torch.cuda.set_device(args.gpu)

    dec = Decoder(args.input, gpu_id=args.gpu, batch_size=args.batch)

    # Encoder needs width/height/fps from metadata
    fps = float(dec.metadata.fps) if dec.metadata.fps else 30.0

    enc = Encoder(
        args.output,
        width=dec.metadata.width,
        height=dec.metadata.height,
        fps=fps,
        codec=args.codec,
        preset=args.preset,
        profile=args.profile,
        qp=args.qp,
        gpu_id=args.gpu,
    )

    out_buf: Optional[torch.Tensor] = None
    frames_total = 0

    try:
        while True:
            frames = dec.read_batch()
            if not frames:
                break

            for frame in frames:
                # Zero-copy wrap GPU memory into a torch Tensor
                t = torch.from_dlpack(frame)

                # Handle either CHW (RGBP) or HWC (RGB) defensively
                if t.ndim == 3 and t.shape[0] == 3:
                    rgbp = _ensure_uint8(t, bit_depth=dec.metadata.bit_depth)
                elif t.ndim == 3 and t.shape[2] == 3:
                    # Interleaved HWC RGB -> CHW RGBP view (no copy) then proceed
                    rgbp = _ensure_uint8(t.permute(2, 0, 1), bit_depth=dec.metadata.bit_depth)
                else:
                    raise RuntimeError(f"Unexpected decoded tensor shape: {tuple(t.shape)} dtype={t.dtype}")

                # Planar RGBP -> packed HWC4 (uint8) on GPU, then encode
                out_buf = rgbp_to_packed(rgbp, out_buf, pack=args.pack)
                enc.encode_frame(out_buf)

                frames_total += 1
                if frames_total % 200 == 0:
                    print(f"[copy] encoded {frames_total} frames...")

        enc.flush()
        print(f"[copy] done. encoded {frames_total} frames.")
        return 0

    finally:
        try:
            enc.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
