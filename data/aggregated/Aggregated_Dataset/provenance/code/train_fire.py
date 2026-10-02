"""Does the thermal branch help? Fire / no-fire patches from ONE Landsat 9 scene (RGB and thermal simultaneous).
Variants: rgb (thermal zeroed) | thermal (RGB zeroed) | both.  Spatial-block 4-fold CV, several seeds.
usage: python train_fire.py <scene_id> <epochs> <seeds>
Results are written after every (variant, seed) so a killed run keeps what finished."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, r"C:\Users\Pranjal\Desktop\SSG\edgeai-digital-model")
from model_architecture import build_model  # noqa: E402

HERE = Path(__file__).parent
SID, EPOCHS, SEEDS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
D = np.load(HERE / "landsat" / SID / "fire_patches.npz")
RGB, THM, Y, RC = D["rgb"], D["thm"], D["y"], D["rc"]
N, K, B = len(Y), 4, 10

# spatial-block folds: 10x10-patch blocks (~38 km), greedily assigned to balance positives
blocks = {}
for i, (r, c) in enumerate(RC):
    blocks.setdefault((r // B, c // B), []).append(i)
order = sorted(blocks.values(), key=lambda idx: -Y[idx].sum())
FOLD = np.zeros(N, int)
load = np.zeros(K)
for idx in order:
    f = int(np.argmin(load + 1e-3 * np.array([(FOLD == k).sum() for k in range(K)])))
    FOLD[idx] = f
    load[f] += Y[idx].sum()
print("N", N, "| positives", Y.sum(), "| fold sizes", [int((FOLD == k).sum()) for k in range(K)],
      "| fold positives", [int(Y[FOLD == k].sum()) for k in range(K)], flush=True)


def batch(idx, T, variant, flip=None):
    x = RGB[idx].astype(np.float32) / 255.0
    t = T[idx].copy()
    if variant == "thermal":
        x = np.zeros_like(x)
    if variant == "rgb":
        t = np.zeros_like(t)
    if flip is not None:
        h, v = flip
        x[h], t[h] = x[h][:, :, ::-1], t[h][:, :, ::-1]
        x[v], t[v] = x[v][:, ::-1], t[v][:, ::-1]
    return x, t


def fit_predict(train, test, variant, seed):
    tf.keras.utils.set_random_seed(seed)
    mu, sd = THM[train].mean(), THM[train].std() + 1e-6
    T = ((THM - mu) / sd)[..., None].astype(np.float32)
    m = build_model()
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    n1 = Y[train].sum(); n0 = len(train) - n1
    cw = {0: len(train) / (2 * n0), 1: len(train) / (2 * n1)}
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        perm = rng.permutation(train)
        for s in range(0, len(perm), 32):
            idx = perm[s:s + 32]
            flip = (rng.random(len(idx)) < 0.5, rng.random(len(idx)) < 0.5)
            xr, xt = batch(idx, T, variant, flip)
            m.train_on_batch([xr, xt], Y[idx], class_weight=cw)
    out = []
    for s in range(0, len(test), 128):
        xr, xt = batch(test[s:s + 128], T, variant)
        out.append(m.predict_on_batch([xr, xt]))
    return np.concatenate(out)[:, 1]


def auc(y, s):
    r = s.argsort().argsort() + 1.0
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def avg_precision(y, s):
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def metrics(y, s):
    o = np.argsort(-s)
    yy = y[o]
    tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
    prec, rec = tp / (tp + fp), tp / y.sum()
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    spec = 1 - fp / (len(y) - y.sum())
    pred = s >= 0.5
    return dict(roc_auc=auc(y, s), ap=avg_precision(y, s), best_f1=float(f1.max()),
                recall_at_spec90=float(rec[spec >= 0.9].max()) if (spec >= 0.9).any() else 0.0,
                f1_at_0p5=float(2 * (pred & (y == 1)).sum() / max(pred.sum() + y.sum(), 1)), base_rate=float(y.mean()))


results, oofs = {}, {}
t0 = time.time()
for seed in range(SEEDS):
    for variant in ("rgb", "thermal", "both"):
        oof = np.full(N, np.nan, np.float32)
        for k in range(K):
            test, train = np.where(FOLD == k)[0], np.where(FOLD != k)[0]
            oof[test] = fit_predict(train, test, variant, seed)
        m = metrics(Y, oof)
        results[f"{variant}|seed{seed}"] = m
        oofs[f"{variant}__seed{seed}"] = oof
        print(f"[{time.time() - t0:5.0f}s] seed{seed} {variant:8s} ROC-AUC {m['roc_auc']:.3f}  AP {m['ap']:.3f}  bestF1 {m['best_f1']:.3f}  recall@spec90 {m['recall_at_spec90']:.3f}", flush=True)
        json.dump(results, open(HERE / "fire_results.json", "w"), indent=1)
        np.savez_compressed(HERE / "fire_oof.npz", **oofs, y=Y, rc=RC, fold=FOLD)
print("DONE", flush=True)
