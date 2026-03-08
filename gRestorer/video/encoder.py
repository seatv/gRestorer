# gRestorer/video/encoder.py
"""
GPU video encoder using NVIDIA PyNvVideoCodec (NVENC), with ffmpeg remux.

Key goals:
- Accept "LADA-like" NVENC knob strings safely (best-effort mapping + fallback).
- Keep defaults high quality but stable across NVENC generations.
- Always remux with audio when possible; keep MP4 compatible (HEVC hvc1, AAC when needed).
- Handle truncated runs (--max-frames): avoid slow-motion by trimming output duration.
- Provide a safe-ish surface for custom mux args (disallow extra -i by default).

Design:
- Encode to a raw elementary stream (.h264/.hevc).
- If output is a container (.mp4/.mkv), remux at close with ffmpeg:
    - video stream copied
    - audio copied/transcoded depending on container compatibility
    - optional subtitles kept (copy for mkv, mov_text for mp4)
    - mp4: optional faststart + timescale
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import shlex
import subprocess

import PyNvVideoCodec as nvc


def _infer_container(output_path: str | Path) -> Optional[str]:
    suf = Path(output_path).suffix.lower()
    if suf in (".mp4", ".m4v", ".mov"):
        return "mp4"
    if suf == ".mkv":
        return "mkv"
    return None


def _raw_ext_for_codec(codec: str) -> str:
    c = codec.lower()
    if c in ("hevc", "h265"):
        return ".hevc"
    if c in ("h264", "avc"):
        return ".h264"
    return ".bit"


def _ffmpeg_input_format(codec: str) -> str:
    c = codec.lower()
    if c in ("hevc", "h265"):
        return "hevc"
    if c in ("h264", "avc"):
        return "h264"
    raise ValueError(f"Unsupported codec for ffmpeg remux: {codec!r}")


def _resolve_ffprobe_path(ffmpeg_path: str) -> str:
    """
    Prefer ffprobe next to ffmpeg if ffmpeg_path is an explicit path;
    otherwise fall back to 'ffprobe' on PATH.
    """
    try:
        p = Path(ffmpeg_path)
        if p.name.lower().startswith("ffmpeg"):
            cand = p.with_name("ffprobe" + p.suffix)
            if cand.exists():
                return str(cand)
    except Exception:
        pass
    return "ffprobe"


def _normalize_preset(p: str) -> str:
    # Accept p7/P7/7 and normalize to P7.
    s = str(p).strip()
    if not s:
        return s
    if s.isdigit():
        return f"P{s}"
    if s.lower().startswith("p") and s[1:].isdigit():
        return f"P{s[1:]}"
    return s.upper()


def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        return int(str(v).strip())
    except Exception:
        return None


def _as_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _parse_kv_style_tokens(tokens: List[str]) -> Dict[str, Any]:
    """
    Parse tokens like:
      -rc constqp -qp 18 -spatial_aq 1
      rc=constqp qp=18
      --rc constqp
    into { "rc":"constqp", "qp":"18", ... }.

    Notes:
    - Flags without values become True.
    - Unknown keys are allowed here; CreateEncoder fallback will handle support.
    """
    out: Dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]

        # key=value form
        if "=" in t and not t.startswith("-"):
            k, v = t.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k:
                out[k] = v
            i += 1
            continue

        # -key value / --key value / -key=value
        if t.startswith("--"):
            key = t[2:]
        elif t.startswith("-"):
            key = t[1:]
        else:
            i += 1
            continue

        if not key:
            i += 1
            continue

        if "=" in key:
            k, v = key.split("=", 1)
            if k:
                out[k] = v
            i += 1
            continue

        # value in next token?
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            out[key] = tokens[i + 1]
            i += 2
        else:
            out[key] = True
            i += 1

    return out


# FFmpeg-style option names -> PyNvVideoCodec-ish names we try.
# (We keep pass-through too; this mapping is additive.)
_FFMPEG_TO_NVC = {
    "rc-lookahead": "lookahead",
    "spatial_aq": "aq",
    "temporal_aq": "temporalaq",
    "tune": "tuning_info",
}


# A broad "known" key set used for a safer retry when allow_unknown=False.
# This is intentionally permissive; unsupported keys are still dropped via fallback.
_KNOWN_KEYS = {
    # core
    "codec", "preset", "profile", "fps",
    "gop", "idrperiod",
    # rc/quality
    "rc", "constqp", "qp", "bitrate", "maxbitrate",
    "vbvbufsize", "vbvinit",
    "multipass", "lookahead",
    "aq", "aq_strength", "temporalaq",
    "tuning_info",
    # b-frames / refs
    "bf", "bframes", "b_ref_mode",
    # misc
    "repeatspspps",
    "maxnumref",
    # timingInfo (some builds accept these)
    "timingInfo::num_unit_in_ticks", "timingInfo::timescale",
    # rate control variants some builds use
    "cbr", "vbr",
}


# Knobs that materially change stream structure / quality policy.
# In strict mode we will not silently drop these to make CreateEncoder succeed.
_STRICT_CRITICAL_KEYS = {
    "codec",
    "preset",
    "profile",
    "fps",
    "gop",
    "idrperiod",
    "bf",
    "bframes",
    "rc",
    "constqp",
    "qp",
    "tuning_info",
}


@dataclass
class _ProbeInfo:
    avg_frame_rate: Optional[str] = None
    r_frame_rate: Optional[str] = None
    duration_sec: Optional[float] = None
    start_time_sec: Optional[float] = None
    audio_codec: Optional[str] = None
    audio_tag: Optional[str] = None
    has_audio: bool = False
    has_subs: bool = False


def _ffprobe_json(path: Path, ffprobe_bin: str) -> dict:
    cmd = [
        ffprobe_bin, "-v", "error",
        "-show_format", "-show_streams",
        "-of", "json",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{r.stderr}")
    return json.loads(r.stdout or "{}")


def _probe_info(path: Path, ffprobe_bin: str) -> _ProbeInfo:
    try:
        j = _ffprobe_json(path, ffprobe_bin)
    except Exception:
        return _ProbeInfo()

    streams = j.get("streams") or []
    fmt = j.get("format") or {}

    pi = _ProbeInfo()

    # duration/start_time
    try:
        if fmt.get("duration") is not None:
            pi.duration_sec = float(fmt["duration"])
    except Exception:
        pi.duration_sec = None
    try:
        if fmt.get("start_time") is not None:
            pi.start_time_sec = float(fmt["start_time"])
    except Exception:
        pi.start_time_sec = None

    v0 = None
    a0 = None
    for s in streams:
        if s.get("codec_type") == "video" and v0 is None:
            v0 = s
        if s.get("codec_type") == "audio" and a0 is None:
            a0 = s
        if s.get("codec_type") == "subtitle":
            pi.has_subs = True

    if v0:
        afr = v0.get("avg_frame_rate")
        rfr = v0.get("r_frame_rate")
        if afr and afr != "0/0":
            pi.avg_frame_rate = str(afr)
        if rfr and rfr != "0/0":
            pi.r_frame_rate = str(rfr)

    if a0:
        pi.has_audio = True
        pi.audio_codec = a0.get("codec_name") or None
        pi.audio_tag = a0.get("codec_tag_string") or None

    return pi


def _rate_to_float(r: Optional[str]) -> Optional[float]:
    if not r or r == "0/0":
        return None
    try:
        return float(Fraction(r))
    except Exception:
        return None


def _is_vfr_like(pi: _ProbeInfo) -> bool:
    # Heuristic: if r_frame_rate differs meaningfully from avg_frame_rate.
    ra = _rate_to_float(pi.avg_frame_rate)
    rr = _rate_to_float(pi.r_frame_rate)
    if ra is None or rr is None:
        return False
    if ra <= 0 or rr <= 0:
        return False
    return abs(rr - ra) / ra > 0.002  # 0.2% threshold


def _sanitize_extra_args(extra: str, output_path: str) -> List[str]:
    """
    Minimal "no footguns" sanitization:
    - Disallow any extra -i (input injection)
    - Disallow tokens that equal the output path
    """
    if not extra:
        return []
    toks = shlex.split(extra)
    bad = {"-i", "--i"}
    for t in toks:
        if t in bad or t.startswith("-i"):
            raise ValueError("mux_extra_args must not include -i / additional inputs")
        if t.strip() == output_path:
            raise ValueError("mux_extra_args must not include the output path")
    return toks


class Encoder:
    """
    Hardware-accelerated video encoder using NVENC (PyNvVideoCodec).

    Expects BGRA memory layout (ARGB word ordering) for input frames.

    New features:
    - encoder mode presets: hq / preview / archive / custom
    - encoder options surfaces:
        * nvenc_options_str: ffmpeg-style "-rc constqp -qp 18 -spatial_aq 1 ..."
        * nvenc_options: dict of KEY->VALUE
      Both merge on top of mode defaults. Unsupported keys are dropped automatically.
    - debug / safety knobs:
        * strict_nvenc: fail if critical NVENC keys would need to be dropped
        * keep_raw: preserve raw elementary stream after remux
        * dump_effective_opts: print the option set that actually succeeded
    - remux controls: mux_audio, mux_keep_subs, mux_extra_args, mp4_faststart
    """

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "hevc",
        preset: str = "P7",
        profile: str = "main",
        qp: int = 20,
        gpu_id: int = 0,
        input_path: str | Path | None = None,
        container: str | None = None,
        ffmpeg_path: str = "ffmpeg",
        # new knobs
        mode: str = "hq",
        nvenc_options_str: str = "",
        nvenc_options: Optional[Dict[str, Any]] = None,
        nvenc_allow_unknown: bool = False,
        mux_audio: str = "auto",  # auto|copy|aac|none
        mux_keep_subs: bool = False,
        mux_extra_args: str = "",
        mp4_faststart: bool = True,
        max_frames: Optional[int] = None,
        strict_nvenc: bool = False,
        keep_raw: bool = False,
        dump_effective_opts: bool = True,
    ) -> None:
        self.output_path = str(output_path)
        self.input_path = str(input_path) if input_path else None
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.codec = str(codec).lower()
        self.preset = _normalize_preset(preset)
        self.profile = str(profile)
        self.qp = int(qp)
        self.gpu_id = int(gpu_id)

        self.ffmpeg_path = str(ffmpeg_path)
        self.ffprobe_path = _resolve_ffprobe_path(self.ffmpeg_path)

        self.mode = str(mode or "hq").lower()
        self.nvenc_options_str = str(nvenc_options_str or "")
        self.nvenc_options = dict(nvenc_options or {})
        self.nvenc_allow_unknown = bool(nvenc_allow_unknown)

        self.mux_audio = str(mux_audio or "auto").lower()
        self.mux_keep_subs = bool(mux_keep_subs)
        self.mux_extra_args = str(mux_extra_args or "")
        self.mp4_faststart = bool(mp4_faststart)
        self.max_frames = int(max_frames) if max_frames is not None else None
        self.strict_nvenc = bool(strict_nvenc)
        self.keep_raw = bool(keep_raw)
        self.dump_effective_opts = bool(dump_effective_opts)

        self._requested_nvenc_opts: Dict[str, Any] = {}
        self._effective_nvenc_opts: Dict[str, Any] = {}
        self._effective_nvenc_stage: Optional[str] = None
        self._effective_nvenc_note: Optional[str] = None

        self.container = container if container is not None else _infer_container(self.output_path)

        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid encoder resolution {self.width}x{self.height}")
        if self.fps <= 0:
            raise ValueError(f"Invalid fps: {self.fps}")

        # Decide where we write bytes
        self._raw_path = self.output_path
        self._needs_remux = False
        if self.container in ("mp4", "mkv"):
            self._needs_remux = True
            self._raw_path = self.output_path + _raw_ext_for_codec(self.codec)

        self._file = open(self._raw_path, "wb")
        self._frames_encoded = 0
        self._closed = False

        # Create encoder
        fmt = "ARGB"  # expects BGRA bytes on little-endian

        enc_opts = self._build_nvenc_options()
        self._requested_nvenc_opts = dict(enc_opts)

        print(f"[Encoder] Creating: {self.output_path}")
        print(f"[Encoder] Resolution: {self.width}x{self.height} @ {self.fps:.3f} fps")
        print(f"[Encoder] Codec: {self.codec}, Preset: {self.preset}, Profile: {self.profile}")
        print(f"[Encoder] Mode: {self.mode}  QP(default)={self.qp}")
        print(
            f"[Encoder] Strict NVENC: {'on' if self.strict_nvenc else 'off'}  "
            f"Keep raw: {'yes' if self.keep_raw else 'no'}"
        )
        if self.nvenc_options_str.strip():
            print(f"[Encoder] nvenc_options_str: {self.nvenc_options_str}")
        if self.nvenc_options:
            print(f"[Encoder] nvenc_options(dict): {self.nvenc_options}")
        print(f"[Encoder] Requested NVENC opts: {enc_opts}")
        print(f"[Encoder] Format: ARGB (expects BGRA memory layout)")
        if self._needs_remux:
            print(f"[Encoder] Container: {self.container} (ffmpeg remux at close)")
            print(f"[Encoder] Raw bitstream: {self._raw_path}")
            if self.input_path:
                print(f"[Encoder] Audio source: {self.input_path}")
        else:
            print("[Encoder] Container: NONE (raw elementary bitstream)")

        self._encoder = self._create_encoder_with_fallback(self.width, self.height, fmt, enc_opts)
        if self.dump_effective_opts and self._effective_nvenc_opts:
            print(
                f"[Encoder] Effective NVENC opts ({self._effective_nvenc_stage or 'unknown'}): "
                f"{self._effective_nvenc_opts}"
            )
            req_keys = set(self._requested_nvenc_opts.keys())
            eff_keys = set(self._effective_nvenc_opts.keys())
            dropped = sorted(req_keys - eff_keys)
            changed = sorted(
                k for k in (req_keys & eff_keys)
                if str(self._requested_nvenc_opts.get(k)) != str(self._effective_nvenc_opts.get(k))
            )
            if dropped:
                print(f"[Encoder] Effective NVENC dropped keys: {dropped}")
            if changed:
                print(
                    "[Encoder] Effective NVENC changed values: "
                    + ", ".join(
                        f"{k}={self._requested_nvenc_opts.get(k)!r}->{self._effective_nvenc_opts.get(k)!r}"
                        for k in changed
                    )
                )
            if self._effective_nvenc_note:
                print(f"[Encoder] Effective NVENC note: {self._effective_nvenc_note}")

        # [CHANGE 4] PTS-derived timing (set by Pipeline before close())
        self._pts_fps: Optional[float] = None
        self._pts_timecodes_path: Optional[str] = None
        self._pts_is_vfr: bool = False

    # -------------------------
    # NVENC option building
    # -------------------------
    def _mode_defaults(self) -> Dict[str, Any]:
        """
        Mode defaults are intentionally conservative; unsupported keys will be dropped.
        You can push to LADA parity by passing nvenc_options_str with -bf/-b_ref_mode/etc.
        """
        gop_frames = max(1, int(round(self.fps * 2.0)))

        if self.mode in ("preview", "fast"):
            return {
                "preset": _normalize_preset(self.preset or "P3"),
                "profile": self.profile,
                "fps": f"{self.fps:g}",
                "gop": str(gop_frames),
                "idrperiod": str(gop_frames),
                "rc": "constqp",
                "constqp": str(max(28, self.qp)),
                "aq": "0",
                "lookahead": "0",
                "bf": "0",
            }

        if self.mode in ("archive", "max", "maxquality"):
            return {
                "preset": _normalize_preset(self.preset or "P7"),
                "profile": self.profile,
                "fps": f"{self.fps:g}",
                "gop": str(gop_frames),
                "idrperiod": str(gop_frames),
                "rc": "constqp",
                "constqp": str(min(18, self.qp)),
                "tuning_info": "high_quality",
                "aq": "1",
                "lookahead": "32",
                # bf/b_ref_mode intentionally not forced; user can opt-in.
            }

        # hq / custom default
        return {
            "preset": _normalize_preset(self.preset or "P7"),
            "profile": self.profile,
            "fps": f"{self.fps:g}",
            "gop": str(gop_frames),
            "idrperiod": str(gop_frames),
            "rc": "constqp",
            "constqp": str(self.qp),
            "tuning_info": "high_quality",
            "aq": "1",
            "lookahead": "32",
            # bf/b_ref_mode are opt-in via nvenc_options_str/dict
        }

    def _merge_options(self, base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(base)
        for k, v in extra.items():
            if v is None:
                continue
            kk = str(k).strip()
            if not kk:
                continue
            out[kk] = v
        return out

    def _build_nvenc_options(self) -> Dict[str, Any]:
        # Required core
        opts: Dict[str, Any] = {
            "codec": self.codec,
            "preset": _normalize_preset(self.preset),
            "profile": self.profile,
            "fps": f"{self.fps:g}",
        }

        # Merge mode defaults
        opts = self._merge_options(opts, self._mode_defaults())

        # Merge dict overrides
        if self.nvenc_options:
            opts = self._merge_options(opts, self.nvenc_options)

        # Merge string overrides (ffmpeg-ish)
        if self.nvenc_options_str.strip():
            toks = shlex.split(self.nvenc_options_str)
            parsed = _parse_kv_style_tokens(toks)

            # Map ffmpeg-style keys into nvc keys too
            mapped: Dict[str, Any] = {}
            for k, v in parsed.items():
                kk = str(k)
                vv = v
                if kk in _FFMPEG_TO_NVC:
                    mapped[_FFMPEG_TO_NVC[kk]] = vv
                mapped[kk] = vv

            opts = self._merge_options(opts, mapped)

        # If user specified -qp but we're in constqp, map qp -> constqp when constqp not explicitly set.
        rc = _as_str(opts.get("rc")) or ""
        if rc.lower() == "constqp":
            if "constqp" not in opts and "qp" in opts:
                opts["constqp"] = opts["qp"]

        # Normalize common fields
        if "preset" in opts:
            opts["preset"] = _normalize_preset(str(opts["preset"]))
        if "bf" in opts and "bframes" not in opts:
            # keep bf as primary; some builds accept bframes instead
            pass

        # Encode expects strings commonly; keep as-is.
        return opts

    def _create_encoder_with_fallback(
        self, w: int, h: int, fmt: str, opts: Dict[str, Any]
    ) -> Any:
        """
        Try CreateEncoder with:
          1) all opts
          2) if allow_unknown=False: filter to _KNOWN_KEYS and retry
          3) iterative drop of optional knobs
          4) last-resort minimal opts (codec/preset/profile/fps)

        In strict mode, critical knobs may not be silently dropped.
        """
        def try_create(o: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Exception]]:
            try:
                return nvc.CreateEncoder(w, h, fmt, False, **o), None
            except Exception as e:
                return None, e

        def record_success(stage: str, used: Dict[str, Any], note: Optional[str] = None) -> None:
            self._effective_nvenc_stage = stage
            self._effective_nvenc_opts = dict(used)
            self._effective_nvenc_note = note

        requested_critical = sorted(k for k in _STRICT_CRITICAL_KEYS if k in opts)

        # Pass 1: all requested opts
        enc, err = try_create(opts)
        if enc is not None:
            record_success("requested", opts)
            return enc

        # Pass 2: known filter (safer)
        if not self.nvenc_allow_unknown:
            filtered = {k: v for (k, v) in opts.items() if k in _KNOWN_KEYS}
            removed = sorted(set(opts.keys()) - set(filtered.keys()))
            removed_critical = sorted(k for k in removed if k in _STRICT_CRITICAL_KEYS)
            if self.strict_nvenc and removed_critical:
                raise RuntimeError(
                    "CreateEncoder failed, and strict NVENC forbids dropping critical keys during known-filter "
                    f"fallback: {removed_critical}. Initial error: {err}"
                )
            enc2, err2 = try_create(filtered)
            if enc2 is not None:
                note = None
                if removed:
                    note = f"dropped unknown/unsupported keys: {removed}"
                    print(f"[Encoder] WARN: dropped unknown opts; CreateEncoder succeeded. (prev err: {err})")
                record_success("known-filtered", filtered, note=note)
                return enc2
            err = err2 or err
            opts = filtered  # continue fallback with filtered set

        # Pass 3: drop optional knobs in a priority order
        drop_order = [
            "b_ref_mode",
            "aq_strength",
            "temporalaq",
            "aq",
            "lookahead",
            "multipass",
            "vbvbufsize",
            "vbvinit",
            "maxbitrate",
            "bitrate",
            "bf",
            "bframes",
            "tuning_info",
            "rc",
            "constqp",
            "qp",
            "idrperiod",
            "gop",
        ]
        o3 = dict(opts)
        dropped_so_far: List[str] = []
        last = err
        for k in drop_order:
            if k not in o3:
                continue
            if self.strict_nvenc and k in _STRICT_CRITICAL_KEYS:
                raise RuntimeError(
                    "CreateEncoder failed, and strict NVENC forbids dropping critical keys. "
                    f"Needed to drop '{k}' to continue fallback. Requested critical keys: {requested_critical}. "
                    f"Last error: {last}"
                )
            o3.pop(k, None)
            dropped_so_far.append(k)
            enc3, err3 = try_create(o3)
            if enc3 is not None:
                print(f"[Encoder] WARN: CreateEncoder succeeded after dropping '{k}'.")
                record_success(
                    "drop-fallback",
                    o3,
                    note=f"dropped fallback keys: {dropped_so_far}",
                )
                return enc3
            last = err3 or last

        # Pass 4: minimal
        minimal = {
            "codec": opts.get("codec", self.codec),
            "preset": opts.get("preset", self.preset),
            "profile": opts.get("profile", self.profile),
            "fps": opts.get("fps", f"{self.fps:g}"),
        }
        if self.strict_nvenc:
            missing_critical = sorted(k for k in requested_critical if k not in minimal)
            if missing_critical:
                raise RuntimeError(
                    "CreateEncoder failed, and strict NVENC forbids falling back to a minimal config that drops "
                    f"critical keys: {missing_critical}. Last error: {last}"
                )

        enc4, err4 = try_create(minimal)
        if enc4 is not None:
            print(f"[Encoder] WARN: CreateEncoder succeeded with minimal opts. (prev err: {last})")
            record_success(
                "minimal",
                minimal,
                note="fell back to minimal opts",
            )
            return enc4

        raise RuntimeError(f"CreateEncoder failed. Last error: {err4 or last}")

    # -------------------------
    # Encode API
    # -------------------------
    def encode_frame(self, frame: Any) -> None:
        if frame is None:
            return
        self._frames_encoded += 1
        bitstream = self._encoder.Encode(frame)
        if bitstream:
            self._file.write(bytearray(bitstream))

    def encode_frames(self, frames: Iterable[Any]) -> None:
        for fr in frames:
            self.encode_frame(fr)

    def flush(self) -> None:
        try:
            tail = self._encoder.EndEncode()
        except Exception as e:
            print(f"[Encoder] ERROR: EndEncode failed: {e}")
            return
        if tail:
            self._file.write(bytearray(tail))
        print(f"[Encoder] Flushing... ({self._frames_encoded} frames submitted)")

    # -------------------------
    # Remux helpers
    # -------------------------
    def _pick_mp4_video_tag(self) -> List[str]:
        if self.container != "mp4":
            return []
        if self.codec in ("hevc", "h265"):
            return ["-tag:v", "hvc1"]
        if self.codec in ("h264", "avc"):
            return ["-tag:v", "avc1"]
        return []

    def _audio_copy_is_mp4_safe(self, codec: Optional[str]) -> bool:
        # Common MP4-safe audio codecs
        return codec in ("aac", "mp3", "alac")

    def _remux_with_ffmpeg(self, input_path: str | None = None) -> None:
        raw_path = Path(self._raw_path)
        out_path = Path(self.output_path)

        if not raw_path.exists():
            print(f"[Encoder] Remux skipped: {raw_path} not found")
            return

        src_path = input_path or self.input_path
        input_video = Path(src_path) if src_path else None

        input_fmt = _ffmpeg_input_format(self.codec)
        mp4_vtag = self._pick_mp4_video_tag()

        have_source = bool(input_video and input_video.exists())
        src_pi = _probe_info(input_video, self.ffprobe_path) if have_source else _ProbeInfo()

        # [CHANGE 4] Prefer PTS-derived fps when available for more accurate remux
        if self._pts_fps is not None and self._pts_fps > 0:
            fps_r = f"{self._pts_fps:g}"
            if have_source and src_pi.avg_frame_rate:
                # Log difference between PTS-derived and metadata fps
                meta_fps = _rate_to_float(src_pi.avg_frame_rate) or self.fps
                if abs(self._pts_fps - meta_fps) / max(meta_fps, 0.001) > 0.002:
                    print(f"[Encoder] Using PTS-derived fps={self._pts_fps:.4f} "
                          f"(metadata={meta_fps:.4f})")
        else:
            fps_r = f"{self.fps:g}"
            if have_source and src_pi.avg_frame_rate:
                fps_r = src_pi.avg_frame_rate

        # Truncation / duration trimming:
        # If output is partial (max_frames or early abort), trim mux to frames_encoded/fps.
        out_t: Optional[str] = None
        fps_f = _rate_to_float(fps_r) or self.fps
        if fps_f > 0 and self._frames_encoded > 0:
            if self.max_frames is not None:
                # explicit partial run
                out_t = f"{(self._frames_encoded / fps_f):.6f}"
            else:
                # heuristic if we can estimate expected frames from duration
                if have_source and src_pi.duration_sec and src_pi.duration_sec > 0:
                    exp_frames = int(round(src_pi.duration_sec * fps_f))
                    if exp_frames > 0 and (self._frames_encoded / exp_frames) < 0.95:
                        out_t = f"{(self._frames_encoded / fps_f):.6f}"

        # VFR-like sources: we can't preserve per-frame timestamps from raw bitstream.
        # Best practice is to mux CFR using avg_frame_rate.
        vfr_like = have_source and _is_vfr_like(src_pi)

        cmd: List[str] = [
            self.ffmpeg_path,
            "-hide_banner",
            "-y",
            "-loglevel", "warning",
            "-fflags", "+genpts",
            "-r", str(fps_r),
            "-f", input_fmt,
            "-i", str(raw_path),
        ]

        if have_source:
            cmd += ["-i", str(input_video)]
            cmd += ["-map", "0:v:0", "-c:v", "copy", *mp4_vtag]

            # Audio policy
            if self.mux_audio == "none":
                cmd += ["-an"]
            else:
                cmd += ["-map", "1:a?"]
                ac = src_pi.audio_codec

                if self.container == "mp4":
                    if self.mux_audio == "aac":
                        cmd += ["-c:a", "aac", "-b:a", "192k"]
                    elif self.mux_audio == "copy":
                        cmd += ["-c:a", "copy"]
                    else:
                        # auto
                        if self._audio_copy_is_mp4_safe(ac):
                            cmd += ["-c:a", "copy"]
                        else:
                            cmd += ["-c:a", "aac", "-b:a", "192k"]

                    # AAC in ADTS -> MP4 requires aac_adtstoasc. Detect using codec_tag_string.
                    if ac == "aac" and (src_pi.audio_tag or "").lower() != "mp4a":
                        # only apply when copying
                        if "-c:a" in cmd:
                            # find last -c:a value
                            for i in range(len(cmd) - 1):
                                if cmd[i] == "-c:a" and cmd[i + 1] == "copy":
                                    cmd += ["-bsf:a", "aac_adtstoasc"]
                                    break
                else:
                    # mkv: copy audio by default
                    if self.mux_audio == "aac":
                        cmd += ["-c:a", "aac", "-b:a", "192k"]
                    elif self.mux_audio == "none":
                        cmd += ["-an"]
                    else:
                        cmd += ["-c:a", "copy"]

            # Subtitle policy
            if self.mux_keep_subs and src_pi.has_subs:
                cmd += ["-map", "1:s?"]
                if self.container == "mp4":
                    # MP4 subtitle compatibility is narrow; mov_text is typical.
                    cmd += ["-c:s", "mov_text"]
                else:
                    cmd += ["-c:s", "copy"]

            # For truncated video, don't let audio extend beyond.
            if out_t is not None:
                cmd += ["-t", out_t, "-shortest"]
        else:
            cmd += ["-an", "-c:v", "copy", *mp4_vtag]
            if out_t is not None:
                cmd += ["-t", out_t]

        # MP4 polish
        if self.container == "mp4":
            if self.mp4_faststart:
                cmd += ["-movflags", "+faststart"]
            cmd += ["-video_track_timescale", "90000"]

        # Extra args (sanitized)
        if self.mux_extra_args.strip():
            cmd += _sanitize_extra_args(self.mux_extra_args, str(out_path))

        cmd += [str(out_path)]

        print("[Encoder] Remux:", " ".join(shlex.quote(x) for x in cmd))
        if have_source:
            print(
                f"[Encoder] Remux source: audio={src_pi.audio_codec or '<none>'} "
                f"subs={'yes' if src_pi.has_subs else 'no'} "
                f"vfr_like={'yes' if vfr_like else 'no'}"
            )
            if out_t is not None:
                print(f"[Encoder] Remux trim: frames={self._frames_encoded} fps={fps_r} t={out_t}s")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Encoder] Remux failed: {e}")

            # MP4 safety retry: force AAC (avoids "no audio" when copy fails).
            if have_source and self.container == "mp4" and self.mux_audio != "none":
                try:
                    cmd2 = [
                        self.ffmpeg_path,
                        "-hide_banner",
                        "-y",
                        "-loglevel", "warning",
                        "-fflags", "+genpts",
                        "-r", str(fps_r),
                        "-f", input_fmt,
                        "-i", str(raw_path),
                        "-i", str(input_video),
                        "-map", "0:v:0",
                        "-c:v", "copy",
                        *mp4_vtag,
                        "-map", "1:a?",
                        "-c:a", "aac",
                        "-b:a", "192k",
                    ]
                    if self.mux_keep_subs and src_pi.has_subs:
                        cmd2 += ["-map", "1:s?", "-c:s", "mov_text"]
                    if out_t is not None:
                        cmd2 += ["-t", out_t, "-shortest"]
                    if self.mp4_faststart:
                        cmd2 += ["-movflags", "+faststart"]
                    cmd2 += ["-video_track_timescale", "90000"]
                    if self.mux_extra_args.strip():
                        cmd2 += _sanitize_extra_args(self.mux_extra_args, str(out_path))
                    cmd2 += [str(out_path)]
                    print("[Encoder] Remux retry (force AAC):", " ".join(shlex.quote(x) for x in cmd2))
                    subprocess.run(cmd2, check=True)
                except Exception as e2:
                    print(f"[Encoder] Remux retry failed: {e2}")
                    return
            else:
                return

        # Delete raw bitstream on success unless explicitly requested otherwise.
        if self.keep_raw:
            print(f"[Encoder] Keeping raw bitstream: {raw_path}")
        else:
            try:
                raw_path.unlink()
            except OSError:
                pass

    # -------------------------
    # Close
    # -------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            self.flush()
        except Exception as e:
            print(f"[Encoder] WARN: flush failed: {e}")

        print(f"[Encoder] Closing output file ({self._frames_encoded} total frames)")
        try:
            try:
                self._file.flush()
            except Exception:
                pass
            self._file.close()
        except Exception:
            pass

        if self._needs_remux:
            self._remux_with_ffmpeg(self.input_path)

    @property
    def frames_encoded(self) -> int:
        return self._frames_encoded

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return (
            f"Encoder(path='{self.output_path}', "
            f"{self.width}x{self.height}, "
            f"{self.codec}, {self._frames_encoded} frames, "
            f"container={self.container!r}, needs_remux={self._needs_remux})"
        )
