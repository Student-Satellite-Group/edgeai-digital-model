"""Score individual patches with (a) the untrained model and (b) out-of-fold trained models, and draw them."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
D = np.load(HERE / "patches.npz")
O = np.load(HERE / "oof.npz")
S = np.load(HERE / "scene_cache.npz")
P = 128
rc = D["rc"]
nd_r, nd_c = (int(v) // P for v in D["nd_rc"])

idx_nd = int(np.where((rc[:, 0] == nd_r) & (rc[:, 1] == nd_c))[0][0])
both = np.where(D["has_thermal"] & D["veg_valid"])[0]
dist = np.hypot(rc[both, 0] - nd_r, rc[both, 1] - nd_c)
idx_b = int(both[dist.argmin()])


def pos_prob(prefix, task, i):
    keys = [k for k in O.files if k.startswith(prefix.replace("|", "__"))]
    return float(np.mean([O[k][i, 1] for k in keys])) if keys else float("nan")


def report(i, title, tasks):
    print(f"\n=== {title}: patch (r,c)={tuple(rc[i])} ===")
    print(f"  ground truth  cloud_frac={D['cloud_frac'][i]:.2f} -> {'CLOUD' if D['y_cloud'][i] else 'no-cloud'}"
          + (f" | veg_frac={D['veg_frac'][i]:.2f} -> {'VEGETATED' if D['y_veg'][i] else 'non-vegetated'}"
             if D['veg_valid'][i] else " | veg label: n/a (patch mostly cloud/other)"))
    print(f"  untrained (random weights)   P(cloud)={O['untrained'][i, 1]:.3f}   P(veg)={O['untrained'][i, 1]:.3f} (same random net)")
    for task, name in tasks:
        lab = "cloud" if task == "cloud" else "vegetated"
        print(f"  {name:32s} P({lab}) = {pos_prob(name_to_key[(task, name)], task, i):.3f}")


name_to_key = {
    ("cloud", "E2 whole-tile, RGB-only"): "E2__cloud",
    ("cloud", "E1 subset, RGB-only"): "E1__cloud__RGB-only",
    ("cloud", "E1 subset, RGB+thermal"): "E1__cloud__RGB+thermal",
    ("veg", "E2 whole-tile, RGB-only"): "E2__veg",
    ("veg", "E1 subset, RGB-only"): "E1__veg__RGB-only",
    ("veg", "E1 subset, RGB+thermal"): "E1__veg__RGB+thermal",
}
cloud_tasks = [("cloud", n) for n in ("E2 whole-tile, RGB-only", "E1 subset, RGB-only", "E1 subset, RGB+thermal")]
veg_tasks = [("veg", n) for n in ("E2 whole-tile, RGB-only", "E1 subset, RGB-only", "E1 subset, RGB+thermal")]

report(idx_nd, "New Delhi (mission AOI)", cloud_tasks)
report(idx_b, "Nearest patch with both labels", cloud_tasks + veg_tasks)

fig, ax = plt.subplots(2, 3, figsize=(11, 7))
for row, (i, ttl) in enumerate(((idx_nd, "New Delhi patch"), (idx_b, "Nearest both-label patch"))):
    r, c = rc[i]
    sl = (slice(r * P, (r + 1) * P), slice(c * P, (c + 1) * P))
    ax[row, 0].imshow(D["rgb"][i]); ax[row, 0].set_title(f"{ttl}: RGB 128x128 @20 m")
    im = ax[row, 1].imshow(D["thm"][i], cmap="inferno", interpolation="nearest")
    ax[row, 1].set_title("Thermal 24x32 (K)"); plt.colorbar(im, ax=ax[row, 1], fraction=0.046)
    ax[row, 2].imshow(S["scl"][sl], cmap="tab20", vmin=0, vmax=11, interpolation="nearest")
    ax[row, 2].set_title("Sentinel-2 SCL (ground truth source)")
    for a in ax[row]:
        a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
plt.savefig(HERE / "single_image.png", dpi=110)
print("\nfigure:", HERE / "single_image.png")
