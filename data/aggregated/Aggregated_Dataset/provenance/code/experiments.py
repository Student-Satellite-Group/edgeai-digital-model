"""Train/evaluate the two-branch model on the real-scene patches (spatially blocked 4-fold CV).

  E1  thermal subset (285 patches): RGB-only (thermal zeroed) vs RGB+thermal, tasks cloud + veg, several seeds
  E2  whole tile   (1764 patches):  RGB-only, tasks cloud + veg
Also reports an UNTRAINED (random-init) reference, and majority-class baselines.
Out-of-fold probabilities are saved so single images can be scored by models that never saw their fold.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, r"C:\Users\Pranjal\Desktop\SSG\edgeai-digital-model")
from model_architecture import build_model  # noqa: E402

HERE = Path(__file__).parent
D = np.load(HERE / "patches.npz")
RGB = D["rgb"].astype(np.float32) / 255.0
THM, RC, HAS_T = D["thm"], D["rc"], D["has_thermal"]
Y = {"cloud": D["y_cloud"], "veg": D["y_veg"]}
VALID = {"cloud": np.ones(len(RGB), bool), "veg": D["veg_valid"]}
K = 4
FOLD = (RC[:, 0] * K) // 42  # spatial row bands


def fit_predict(train, test, y, use_thermal, epochs, seed):
    tf.keras.utils.set_random_seed(seed)
    if use_thermal:
        mu, sd = np.nanmean(THM[train]), np.nanstd(THM[train])
        T = np.nan_to_num((THM - mu) / sd, nan=0.0)[..., None].astype(np.float32)
    else:
        T = np.zeros((len(RGB), 24, 32, 1), np.float32)
    m = build_model()
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    n1 = y[train].sum(); n0 = len(train) - n1
    cw = {0: len(train) / (2 * max(n0, 1)), 1: len(train) / (2 * max(n1, 1))}
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        hf, vf = rng.random(len(train)) < 0.5, rng.random(len(train)) < 0.5
        xr, xt = RGB[train].copy(), T[train].copy()
        xr[hf], xt[hf] = xr[hf][:, :, ::-1], xt[hf][:, :, ::-1]
        xr[vf], xt[vf] = xr[vf][:, ::-1], xt[vf][:, ::-1]
        m.fit([xr, xt], y[train], epochs=1, batch_size=32, verbose=0, class_weight=cw)
    return m.predict([RGB[test], T[test]], batch_size=128, verbose=0)


def cv(mask, y, use_thermal, epochs, seed):
    oof = np.full((len(RGB), 2), np.nan, np.float32)
    idx = np.where(mask)[0]
    for k in range(K):
        test, train = idx[FOLD[idx] == k], idx[FOLD[idx] != k]
        if len(test) == 0 or len(np.unique(y[train])) < 2:
            continue
        oof[test] = fit_predict(train, test, y, use_thermal, epochs, seed)
    return oof


def metrics(y, prob):
    ok = np.isfinite(prob[:, 0])
    y, pred = y[ok], prob[ok].argmax(1)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    return dict(n=int(ok.sum()), acc=(tp + tn) / max(len(y), 1), majority=max(y.mean(), 1 - y.mean()),
                bal_acc=(rec + spec) / 2, precision=prec, recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-9), tp=tp, fp=fp, fn=fn, tn=tn)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        t0 = time.time()
        idx = np.where(VALID["cloud"])[0]
        te, tr = idx[FOLD[idx] == 1], idx[FOLD[idx] != 1]
        fit_predict(tr, te, Y["cloud"], False, 2, 0)
        print(f"bench: 2 epochs on {len(tr)} train patches = {time.time() - t0:.1f}s (incl. TF startup)")
        sys.exit()

    epochs_e1, epochs_e2, seeds_e1 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    res, oofs = {}, {}
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)

    # untrained reference (random init, no training at all)
    tf.keras.utils.set_random_seed(0)
    m = build_model()
    T0 = np.zeros((len(RGB), 24, 32, 1), np.float32)
    p0 = m.predict([RGB, T0], batch_size=128, verbose=0)
    for task in ("cloud", "veg"):
        mk = VALID[task]
        res[f"untrained|{task}"] = metrics(Y[task][mk], p0[mk])
    oofs["untrained"] = p0
    log("untrained reference done")

    for task in ("cloud", "veg"):                          # E2: whole tile, RGB only
        mk = VALID[task]
        oof = cv(mk, Y[task], False, epochs_e2, 0)
        res[f"E2 whole-tile RGB-only|{task}"] = metrics(Y[task][mk], oof[mk])
        oofs[f"E2|{task}"] = oof
        log(f"E2 {task}: {res[f'E2 whole-tile RGB-only|{task}']}")

    for task in ("cloud", "veg"):                          # E1: thermal subset ablation
        mk = VALID[task] & HAS_T
        for use_t in (False, True):
            name = "RGB+thermal" if use_t else "RGB-only"
            agg = []
            for seed in range(seeds_e1):
                oof = cv(mk, Y[task], use_t, epochs_e1, seed)
                agg.append(metrics(Y[task][mk], oof[mk]))
                oofs[f"E1|{task}|{name}|seed{seed}"] = oof
                log(f"E1 {task} {name} seed{seed}: acc={agg[-1]['acc']:.3f} f1={agg[-1]['f1']:.3f}")
            res[f"E1 thermal-subset {name}|{task}"] = {
                "per_seed": agg,
                "acc_mean": float(np.mean([a["acc"] for a in agg])), "acc_std": float(np.std([a["acc"] for a in agg])),
                "f1_mean": float(np.mean([a["f1"] for a in agg])), "majority": agg[0]["majority"], "n": agg[0]["n"]}

    json.dump(res, open(HERE / "results.json", "w"), indent=1)
    np.savez_compressed(HERE / "oof.npz", **{k.replace("|", "__"): v for k, v in oofs.items()})
    log("DONE")
