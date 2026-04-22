# gRestorer/restorer/face_swap_helpers.py

from __future__ import annotations

import cv2
import numpy as np


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def gaussian_blur_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(0.8, float(sigma))
    out = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.ascontiguousarray(np.clip(out, 0.0, 1.0))


def draw_soft_ellipse(
    mask: np.ndarray,
    center: tuple[float, float],
    axes: tuple[float, float],
    value: float = 1.0,
) -> None:
    cx, cy = int(round(center[0])), int(round(center[1]))
    ax = max(1, int(round(axes[0])))
    ay = max(1, int(round(axes[1])))
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0.0, 0.0, 360.0, float(value), thickness=-1)


def draw_soft_rect(
    mask: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    value: float = 1.0,
) -> None:
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    cv2.rectangle(mask, (x0, y0), (x1, y1), float(value), thickness=-1)


def build_broad_destination_mask(
    aligned_size: int,
    target_landmarks_aligned: np.ndarray,
) -> np.ndarray:
    size = int(aligned_size)
    lm = np.asarray(target_landmarks_aligned, dtype=np.float32)

    dst = np.zeros((size, size), dtype=np.float32)

    if lm.shape[0] < 5:
        draw_soft_ellipse(
            dst,
            center=(size * 0.50, size * 0.58),
            axes=(size * 0.34, size * 0.43),
            value=1.0,
        )
        dst = gaussian_blur_mask(dst, size / 34.0)
        return dst

    left_eye, right_eye, nose, left_mouth, right_mouth = lm[:5]
    eye_mid = (left_eye + right_eye) * 0.5
    mouth_mid = (left_mouth + right_mouth) * 0.5

    eye_dist = max(1.0, float(np.linalg.norm(right_eye - left_eye)))
    mouth_width = max(1.0, float(np.linalg.norm(right_mouth - left_mouth)))
    eye_to_mouth = max(1.0, float(np.linalg.norm(mouth_mid - eye_mid)))
    eye_to_nose = max(1.0, float(np.linalg.norm(nose - eye_mid)))
    nose_to_mouth = max(1.0, float(np.linalg.norm(mouth_mid - nose)))

    yaw_proxy = clip01(abs(float(nose[0] - eye_mid[0])) / (0.45 * eye_dist))
    frontalness = 1.0 - yaw_proxy

    cx = float(0.30 * nose[0] + 0.70 * mouth_mid[0])
    cy = float(0.22 * eye_mid[1] + 0.78 * mouth_mid[1])

    draw_soft_ellipse(
        dst,
        center=(cx, cy + 0.02 * eye_dist),
        axes=(
            max(0.95 * eye_dist, 0.90 * mouth_width),
            max(1.60 * eye_to_mouth, 1.15 * nose_to_mouth),
        ),
        value=1.0,
    )

    draw_soft_ellipse(
        dst,
        center=(mouth_mid[0], mouth_mid[1] + 0.42 * nose_to_mouth),
        axes=(0.82 * mouth_width, 0.62 * eye_to_mouth),
        value=1.0,
    )

    draw_soft_ellipse(
        dst,
        center=(left_eye[0] - 0.22 * eye_dist, nose[1] + 0.34 * eye_to_nose),
        axes=(0.32 * eye_dist, 0.48 * eye_to_mouth),
        value=1.0,
    )
    draw_soft_ellipse(
        dst,
        center=(right_eye[0] + 0.22 * eye_dist, nose[1] + 0.34 * eye_to_nose),
        axes=(0.32 * eye_dist, 0.48 * eye_to_mouth),
        value=1.0,
    )

    draw_soft_ellipse(
        dst,
        center=(eye_mid[0], eye_mid[1] - 0.12 * eye_to_nose),
        axes=(0.52 * eye_dist, 0.28 * eye_to_mouth),
        value=0.85,
    )

    draw_soft_rect(
        dst,
        (nose[0] - 0.22 * eye_dist, eye_mid[1] + 0.12 * eye_to_nose),
        (nose[0] + 0.22 * eye_dist, mouth_mid[1] + 0.10 * nose_to_mouth),
        value=1.0,
    )

    y = np.arange(size, dtype=np.float32)
    top_zero = eye_mid[1] - (0.45 - 0.08 * yaw_proxy) * eye_to_nose
    top_one = eye_mid[1] + (0.30 + 0.10 * frontalness) * eye_to_nose
    top_ramp = np.clip((y - top_zero) / max(1.0, (top_one - top_zero)), 0.0, 1.0)
    dst *= top_ramp[:, None]

    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    side_core = 1.0 - np.abs(x - 0.5) / 0.5
    side_min = 0.02 + 0.10 * yaw_proxy
    side = np.clip((side_core - side_min) / max(1e-6, (1.0 - side_min)), 0.0, 1.0)
    dst *= side[None, :]

    dst = gaussian_blur_mask(dst, size / 30.0)
    return np.ascontiguousarray(np.clip(dst, 0.0, 1.0))


def build_source_mask_from_backend(
    backend_mask_f32: np.ndarray | None,
    aligned_size: int,
    target_landmarks_aligned: np.ndarray,
) -> np.ndarray:
    size = int(aligned_size)

    if backend_mask_f32 is None:
        src = np.zeros((size, size), dtype=np.float32)
    else:
        src = np.asarray(backend_mask_f32, dtype=np.float32)
        if src.ndim == 3:
            src = src[..., 0]
        if src.shape[:2] != (size, size):
            src = cv2.resize(src, (size, size), interpolation=cv2.INTER_LINEAR)
        src = np.clip(src, 0.0, 1.0)

    lm = np.asarray(target_landmarks_aligned, dtype=np.float32)
    if lm.shape[0] < 5:
        src = gaussian_blur_mask(src, size / 44.0)
        return np.ascontiguousarray(np.clip(src, 0.0, 1.0))

    left_eye, right_eye, nose, left_mouth, right_mouth = lm[:5]
    eye_mid = (left_eye + right_eye) * 0.5
    mouth_mid = (left_mouth + right_mouth) * 0.5

    eye_dist = max(1.0, float(np.linalg.norm(right_eye - left_eye)))
    mouth_width = max(1.0, float(np.linalg.norm(right_mouth - left_mouth)))
    eye_to_mouth = max(1.0, float(np.linalg.norm(mouth_mid - eye_mid)))
    nose_to_mouth = max(1.0, float(np.linalg.norm(mouth_mid - nose)))

    support = np.zeros((size, size), dtype=np.float32)

    draw_soft_ellipse(
        support,
        center=(mouth_mid[0], mouth_mid[1] + 0.18 * nose_to_mouth),
        axes=(0.60 * mouth_width, 0.42 * eye_to_mouth),
        value=1.0,
    )
    draw_soft_rect(
        support,
        (nose[0] - 0.16 * eye_dist, nose[1] + 0.10 * nose_to_mouth),
        (nose[0] + 0.16 * eye_dist, mouth_mid[1] + 0.12 * nose_to_mouth),
        value=1.0,
    )
    support = gaussian_blur_mask(support, size / 42.0)

    src = np.maximum(src, 0.45 * support)
    src = gaussian_blur_mask(src, size / 48.0)
    return np.ascontiguousarray(np.clip(src, 0.0, 1.0))