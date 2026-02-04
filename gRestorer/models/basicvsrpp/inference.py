# gRestorer/gRestorer/models/basicvsrpp/inference.py
from __future__ import annotations

from typing import Any, Dict, Tuple
import os
import torch

from gRestorer.models.basicvsrpp.lada.basicvsr_plusplus_net import BasicVSRPlusPlusNet


def get_default_gan_inference_config() -> dict:
    """
    Match LADA stage2-ish defaults (generator only for inference).
    """
    return dict(
        generator=dict(
            mid_channels=64,
            num_blocks=15,
            max_residue_magnitude=10,
            spynet_pretrained=None,
        )
    )


def _load_checkpoint_blob(checkpoint_path: str) -> Dict[str, Any]:
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")


def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    # Common patterns
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    # Sometimes it *is* already a state_dict
    if all(isinstance(k, str) for k in ckpt.keys()):
        # best-effort: looks like a raw state_dict
        return ckpt  # type: ignore[return-value]
    raise ValueError("Could not locate a usable state_dict in checkpoint.")


def _pick_and_strip_prefix(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], str]:
    """
    Prefer EMA generator weights when available (LADA uses generator_ema at inference).
    Returns (stripped_state_dict, prefix_used).
    """
    prefixes = [

        # Prefer EMA weights if present
        "generator_ema.",
        "module.generator_ema.",
        "ema_model.generator.",
        "module.ema_model.generator.",

        # Then non-EMA generator
        "generator.",
        "module.generator.",
        "net_g.",
        "module.net_g.",
        "model.generator.",
        "module.model.generator.",
        "module.",  # last resort
    ]

    for p in prefixes:
        if any(k.startswith(p) for k in sd.keys()):
            stripped = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
            if stripped:
                return stripped, p

    # Nothing matched — return as-is (might already be generator-only)
    return sd, "<none>"


class _BasicVSRPPWrapper(torch.nn.Module):
    """
    Minimal wrapper so callers can do:
        out = model(inputs=BTCHW)
    """
    def __init__(self, generator: torch.nn.Module):
        super().__init__()
        self.generator = generator

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.generator(inputs)


def load_model(
    config: str | dict | None,
    checkpoint_path: str,
    device: torch.device | str,
    fp16: bool = False,
) -> torch.nn.Module:
    if isinstance(device, str):
        device = torch.device(device)

    # Config handling
    if config is None:
        config = get_default_gan_inference_config()
    if isinstance(config, str):
        scope: Dict[str, Any] = {}
        with open(config, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, config, "exec"), scope, scope)
        if "model" in scope and isinstance(scope["model"], dict):
            config = scope["model"]
        elif "config" in scope and isinstance(scope["config"], dict):
            config = scope["config"]
        else:
            raise ValueError(f"Config file {config!r} did not define dict `model` or `config`.")
    if not isinstance(config, dict):
        raise TypeError("config must be a dict, a config.py path, or None")

    gen_cfg = config.get("generator")
    if not isinstance(gen_cfg, dict):
        raise ValueError("config must contain dict `generator`")

    generator = BasicVSRPlusPlusNet(
        mid_channels=int(gen_cfg.get("mid_channels", 64)),
        num_blocks=int(gen_cfg.get("num_blocks", 15)),
        max_residue_magnitude=int(gen_cfg.get("max_residue_magnitude", 10)),
        spynet_pretrained=gen_cfg.get("spynet_pretrained", None),
    )

    ckpt = _load_checkpoint_blob(checkpoint_path)
    sd = _extract_state_dict(ckpt)
    sd, used_prefix = _pick_and_strip_prefix(sd)

    # Load weights (strict by default; allow override)
    allow_partial = os.environ.get("GR_ALLOW_PARTIAL_WEIGHTS", "").strip() in ("1", "true", "True")

    missing, unexpected = generator.load_state_dict(sd, strict=False)

    # Loud, useful diagnostics
    total_keys = len(generator.state_dict().keys())
    print(f"[BasicVSR++] checkpoint prefix used: {used_prefix}")
    print(f"[BasicVSR++] load_state_dict: missing={len(missing)}/{total_keys} unexpected={len(unexpected)}")

    if (missing or unexpected) and not allow_partial:
        # Show a tiny sample to avoid log spam
        m_samp = "\n".join(missing[:20])
        u_samp = "\n".join(unexpected[:20])
        raise RuntimeError(
            "BasicVSR++ weights did not load cleanly.\n"
            "This will produce garbage / no-op restoration.\n\n"
            f"Prefix used: {used_prefix}\n"
            f"Missing keys (first 20):\n{m_samp if m_samp else '(none)'}\n\n"
            f"Unexpected keys (first 20):\n{u_samp if u_samp else '(none)'}\n\n"
            "If you *really* want to run with partial weights, set:\n"
            "  GR_ALLOW_PARTIAL_WEIGHTS=1"
        )

    model = _BasicVSRPPWrapper(generator=generator).to(device).eval()
    if fp16 and device.type == "cuda":
        model = model.half()
    return model
