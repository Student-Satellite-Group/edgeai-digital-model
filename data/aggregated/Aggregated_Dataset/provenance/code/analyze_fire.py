"""Deeper analysis of the fire-detection experiment: CIs, per-fold, operating points, untrained baselines, dose-response."""
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SID = "LC09_L2SP_158014_20260817_02_T1"
O = np.load(HERE / "fire_oof.npz")
P = np.load(HERE / "landsat" / SID / "fire_patches.npz")
y, rc, fold = O["y"], O["rc"], O["fold"]
thm, rgb, nfire = P["thm"], P["rgb"], P["nfire"]
N = len(y)
blk = (rc[:, 0] // 10) * 100 + (rc[:, 1] // 10)
ub = np.unique(blk)
rng = np.random.default_rng(0)


def auc(yy, s):
    r = s.argsort().argsort() + 1.0
    n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ap(yy, s):
    o = np.argsort(-s, kind="stable"); yo = yy[o]
    return float((np.cumsum(yo) / np.arange(1, len(yo) + 1) * yo).sum() / yo.sum())


def opoint(yy, s, spec=0.9):
    thr = np.quantile(s[yy == 0], spec)
    pred = s > thr
    tp = int((pred & (yy == 1)).sum()); fp = int((pred & (yy == 0)).sum())
    return dict(thr=float(thr), tp=tp, fp=fp, fn=int(yy.sum() - tp), tn=int((yy == 0).sum() - fp),
                recall=tp / yy.sum(), precision=tp / max(tp + fp, 1))


variants = ("rgb", "thermal", "both")
seedavg = {v: np.mean([O[f"{v}__seed{s}"] for s in range(3)], axis=0) for v in variants}
per_seed = {v: [O[f"{v}__seed{s}"] for s in range(3)] for v in variants}

print("== 1. Seed-averaged out-of-fold scores (average of the 3 seeds' probabilities) ==")
for v in variants:
    print(f"  {v:8s} ROC-AUC {auc(y, seedavg[v]):.3f} | AP {ap(y, seedavg[v]):.3f}")

print("\n== 2. Operating point at 90% specificity (seed-averaged) ==")
for v in variants:
    o = opoint(y, seedavg[v])
    print(f"  {v:8s} TP {o['tp']:3d} FN {o['fn']:3d} FP {o['fp']:3d} TN {o['tn']:4d} | recall {o['recall']:.3f} | precision {o['precision']:.3f}")

print("\n== 3. Per-fold ROC-AUC / AP (mean over 3 seeds); fold n / positives ==")
for k in range(4):
    m = fold == k
    row = f"  fold {k} n={m.sum():4d} pos={y[m].sum():2d} | "
    for v in variants:
        a = np.mean([auc(y[m], per_seed[v][s][m]) for s in range(3)]); p = np.mean([ap(y[m], per_seed[v][s][m]) for s in range(3)])
        row += f"{v}: {a:.3f}/{p:.3f}  "
    print(row)

print("\n== 4. Paired per-seed comparison (thermal-only minus fused) ==")
for s in range(3):
    print(f"  seed {s}: dROC-AUC {auc(y, per_seed['thermal'][s]) - auc(y, per_seed['both'][s]):+.3f} | dAP {ap(y, per_seed['thermal'][s]) - ap(y, per_seed['both'][s]):+.3f}")

# simple untrained statistics
T = thm.reshape(N, -1)
stats = {
    "thermal contrast (max - median)": T.max(1) - np.median(T, 1),
    "thermal max": T.max(1),
    "thermal p95 - median": np.percentile(T, 95, axis=1) - np.median(T, 1),
    "thermal std": T.std(1),
    "thermal mean": T.mean(1),
}
R = rgb.reshape(N, -1, 3).astype(np.float32)
stats.update({"RGB mean brightness": R.mean((1, 2)), "RGB red mean": R[..., 0].mean(1), "RGB green mean": R[..., 1].mean(1),
              "RGB blue mean": R[..., 2].mean(1), "RGB std": R.mean(2).std(1)})
print("\n== 5. Untrained single-number statistics (no fitting; same 1,848 patches) ==")
for k, s in stats.items():
    print(f"  {k:34s} ROC-AUC {auc(y, s):.3f} | AP {ap(y, s):.3f}")

# block bootstrap
def boot(fn, B=1000):
    out = []
    for _ in range(B):
        pick = rng.choice(ub, len(ub), replace=True)
        idx = np.concatenate([np.where(blk == b)[0] for b in pick])
        if y[idx].sum() < 5:
            continue
        out.append(fn(idx))
    return np.array(out)


print(f"\n== 6. Spatial-block bootstrap (resampling the {len(ub)} 38-km blocks, B=1000), 95% intervals ==")
ci = lambda a: f"[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"
for v in variants:
    a = boot(lambda i, v=v: auc(y[i], seedavg[v][i])); p = boot(lambda i, v=v: ap(y[i], seedavg[v][i]))
    print(f"  {v:8s} ROC-AUC {auc(y, seedavg[v]):.3f} {ci(a)} | AP {ap(y, seedavg[v]):.3f} {ci(p)}")
s = stats["thermal contrast (max - median)"]
a = boot(lambda i: auc(y[i], s[i])); p = boot(lambda i: ap(y[i], s[i]))
print(f"  contrast stat ROC-AUC {auc(y, s):.3f} {ci(a)} | AP {ap(y, s):.3f} {ci(p)}")
print("  paired differences (same resamples):")
pairs = (("thermal", "both"), ("thermal", "rgb"), ("both", "rgb"))
for a_, b_ in pairs:
    d1, d2 = [], []
    rng2 = np.random.default_rng(1)
    for _ in range(1000):
        pick = rng2.choice(ub, len(ub), replace=True)
        idx = np.concatenate([np.where(blk == b)[0] for b in pick])
        if y[idx].sum() < 5:
            continue
        d1.append(auc(y[idx], seedavg[a_][idx]) - auc(y[idx], seedavg[b_][idx]))
        d2.append(ap(y[idx], seedavg[a_][idx]) - ap(y[idx], seedavg[b_][idx]))
    d1, d2 = np.array(d1), np.array(d2)
    print(f"    {a_} - {b_}: dROC-AUC {np.mean(d1):+.3f} {ci(d1)} (frac>0 {np.mean(d1 > 0):.2f}) | dAP {np.mean(d2):+.3f} {ci(d2)} (frac>0 {np.mean(d2 > 0):.2f})")
s = stats["thermal contrast (max - median)"]
d1 = []
rng2 = np.random.default_rng(2)
for _ in range(1000):
    pick = rng2.choice(ub, len(ub), replace=True)
    idx = np.concatenate([np.where(blk == b)[0] for b in pick])
    if y[idx].sum() < 5:
        continue
    d1.append(auc(y[idx], s[idx]) - auc(y[idx], seedavg["thermal"][idx]))
d1 = np.array(d1)
print(f"    contrast stat - thermal CNN: dROC-AUC {np.mean(d1):+.3f} {ci(d1)} (frac>0 {np.mean(d1 > 0):.2f})")

print("\n== 7. Dose-response: thermal contrast (K) by number of SWIR fire pixels in the patch ==")
c = stats["thermal contrast (max - median)"]; mx = T.max(1)
for lo, hi, name in ((0, 0, "0 (negative)"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 15, "8-15"), (16, 63, "16-63"), (64, 99999, "64+")):
    m = (nfire >= lo) & (nfire <= hi)
    if m.sum():
        print(f"  {name:13s} n={m.sum():4d} | contrast median {np.median(c[m]):6.2f} K (p25 {np.percentile(c[m], 25):5.2f}, p75 {np.percentile(c[m], 75):5.2f}) | max-temp median {np.median(mx[m]):6.1f} K")
print("  fire-pixel count in positive patches: median %d, p90 %d, max %d" % (np.median(nfire[y == 1]), np.percentile(nfire[y == 1], 90), nfire.max()))

print("\n== 8. Where does the thermal CNN disagree with the contrast statistic? ==")
thr_c = np.quantile(c[y == 0], 0.9)
cs_hit = (c > thr_c) & (y == 1)
th_hit = (seedavg["thermal"] > np.quantile(seedavg["thermal"][y == 0], 0.9)) & (y == 1)
print("  at 90%% specificity: contrast stat finds %d/%d, thermal CNN finds %d/%d, both %d, CNN-only %d, stat-only %d"
      % (cs_hit.sum(), y.sum(), th_hit.sum(), y.sum(), (cs_hit & th_hit).sum(), (th_hit & ~cs_hit).sum(), (cs_hit & ~th_hit).sum()))
print("  base rate %.4f | positives %d | negatives %d | blocks %d" % (y.mean(), y.sum(), (y == 0).sum(), len(ub)))
