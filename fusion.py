"""
fusion.py — Task 4.5: weighted overlay fusion of optical (RGB) and registered
thermal imagery.

    F(x, y) = alpha * RGB(x, y) + (1 - alpha) * Thermal_registered(x, y)

Role update (post Task 5.2 revision, see task_definition.md): this pixel-level
blend is no longer the classifier's input. The trained model
(model_architecture.py) fuses RGB and native-resolution thermal at the
*feature* level instead, via two separate branches -- avoiding the need to
warp the native 32x24 thermal grid onto RGB's full pixel grid before any
learning happens. fuse_images()/alpha_sweep() below remain valid as a
standalone data product (Architecture Section 7.3, "fused output frame") for
logging/visualization, just decoupled from inference.

Both inputs are normalised to [0, 1] before blending, then clipped back to
uint8. `fuse_images()` performs a single blend; `alpha_sweep()` blends over a
range of alpha values and reports SSIM/PSNR against a probe image for each
alpha (so a recommended alpha can be picked later, Task 4.6 / Phase-4 report).

Placeholder check baked into __main__: alpha=0 -> pure thermal,
alpha=1 -> pure optical.
"""

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def _to_float01(image):
    """Return a float [0, 1] 3-channel image for blending.

    Thermal input may be 2-D grayscale; it is stacked to 3 channels."""
    if image.ndim == 2:
        image = cv2.merge([image] * 3)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    return image.astype(np.float32) / 255.0


def _back_to_uint8(img01):
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)


def fuse_images(optical, thermal_registered, alpha=0.5):
    """Blend normalised optical and registered-thermal images by alpha.

    alpha=0 -> pure thermal, alpha=1 -> pure optical. Returns a uint8 3-channel
    image of shape (H, W, 3).
    """
    opt = _to_float01(optical)
    thm = _to_float01(thermal_registered)
    fused = alpha * opt + (1.0 - alpha) * thm
    return _back_to_uint8(fused)


def alpha_sweep(optical, thermal_registered, alpha_min=0.0, alpha_max=1.0,
                step=0.1, probe=None):
    """Blend across the alpha range, returning (alphas, results).

    If `probe` (an image) is given, SSIM/PSNR vs probe are included in each
    result row for quantitative comparison.
    """
    alphas = np.arange(alpha_min, alpha_max + 1e-9, step)
    results = []
    for a in alphas:
        fused = fuse_images(optical, thermal_registered, float(a))
        row = {"alpha": float(a), "fused": fused}
        if probe is not None:
            probe_u8 = _back_to_uint8(_to_float01(probe))
            row["ssim"] = ssim(fused, probe_u8, channel_axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                row["psnr"] = psnr(fused, probe_u8)  # inf when identical
        results.append(row)
    return list(alphas), results


def _synthetic_pair(seed=1, size=256):
    """Placeholder optical + thermal frames with distinct patterns."""
    rng = np.random.default_rng(seed)
    optical = np.zeros((size, size, 3), np.uint8)
    cv2.rectangle(optical, (30, 30), (120, 120), (40, 180, 40), -1)
    cv2.circle(optical, (170, 180), 40, (200, 60, 30), -1)
    thermal = np.full((size, size), 60, np.uint8)
    cv2.circle(thermal, (170, 180), 40, 230, -1)
    cv2.rectangle(thermal, (30, 30), (120, 120), 120, -1)
    return optical, thermal


if __name__ == "__main__":
    opt, thm = _synthetic_pair()

    f0 = fuse_images(opt, thm, alpha=0.0)
    f1 = fuse_images(opt, thm, alpha=1.0)

    assert np.array_equal(f0, cv2.merge([thm] * 3)), "alpha=0 must be pure thermal"
    assert np.array_equal(f1, opt), "alpha=1 must be pure optical"
    print("PASS: alpha=0 -> pure thermal, alpha=1 -> pure optical")

    alphas, rows = alpha_sweep(opt, thm, probe=opt)
    print(f"alpha sweep: {len(rows)} steps "
          f"({alphas[0]:.1f} -> {alphas[-1]:.1f})")
    for r in rows:
        print(f"  alpha={r['alpha']:.1f}  SSIM={r['ssim']:.4f}  "
              f"PSNR={r['psnr']:.2f} dB")
    assert len(rows) == 11, "alpha_sweep should return 0.0..1.0 step 0.1"
    print("PASS: SSIM/PSNR computed for every alpha")