"""Image preprocessing (MSR aspect-preserving pad) + analog-tuned augmentation."""
from __future__ import annotations

import random
from functools import partial

import cv2
import numpy as np


def fit_pad(image, target_h: int = 48, target_w: int = 320, **_):
    """Aspect-preserving resize into target_h x target_w (MSR), black-padded.

    Keeps digit shapes and the decimal/red-wheel boundary undistorted, unlike a
    plain stretch. Drum backgrounds are dark, so black padding blends in.
    """
    h, w = image.shape[:2]
    s = min(target_h / h, target_w / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interp)
    ch = image.shape[2] if image.ndim == 3 else 1
    canvas = np.zeros((target_h, target_w, ch), dtype=image.dtype)
    y0, x0 = (target_h - nh) // 2, (target_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized.reshape(nh, nw, ch)
    return canvas


def synthetic_rolling(image, **_):
    """Simulate a wheel caught mid-rotation: roll one digit-wide column vertically.

    Produces the half-of-two-digits look that is the signature hard case, letting
    us oversample it cheaply during training.
    """
    h, w = image.shape[:2]
    bw = max(4, w // random.randint(5, 9))
    x0 = random.randint(0, max(0, w - bw))
    shift = random.randint(h // 4, max(h // 4 + 1, h // 2)) * random.choice([-1, 1])
    out = image.copy()
    out[:, x0:x0 + bw] = np.roll(out[:, x0:x0 + bw], shift, axis=0)
    return out


def _safe(make):
    try:
        return make()
    except Exception:
        return None


def build_transforms(img_h: int, img_w: int, training: bool = True,
                     aug_level: str = "analog", rolling_aug: float = 0.15):
    """Build an albumentations pipeline. Never flips (digit orientation is fixed)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ops = [A.Lambda(image=partial(fit_pad, target_h=img_h, target_w=img_w), name="fit_pad")]

    if training and aug_level != "none":
        if rolling_aug > 0:
            ops.append(A.Lambda(image=synthetic_rolling, name="rolling", p=rolling_aug))
        candidates = [
            lambda: A.Affine(scale=(0.9, 1.08), rotate=(-5, 5), shear=(-4, 4),
                             translate_percent=(-0.03, 0.03), fill=0, p=0.5),
            lambda: A.Perspective(scale=(0.02, 0.06), p=0.3),
            lambda: A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
            lambda: A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            lambda: A.OneOf([A.MotionBlur(blur_limit=7), A.GaussianBlur(blur_limit=(3, 7)),
                             A.Defocus(radius=(1, 3))], p=0.3),
            lambda: A.GaussNoise(p=0.2),
            lambda: A.ImageCompression(quality_range=(40, 90), p=0.3),
            lambda: A.CoarseDropout(p=0.3),
            lambda: A.RandomShadow(p=0.15),
        ]
        if aug_level == "light":
            candidates = candidates[:4]
        ops.extend([t for t in (_safe(c) for c in candidates) if t is not None])

    ops.extend([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()])
    return A.Compose(ops)
