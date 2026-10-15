"""Offline analysis of the Kaggle outputs: metrics, calibration, uncertainty, selective prediction."""
import json, glob, os, sys
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DATA", os.path.join(REPO, "data"))        # folder with the .h5 files
RESULTS = os.environ.get("RESULTS", os.path.join(REPO, "results"))  # csv / npy output
FIGS = os.environ.get("FIGS", os.path.join(REPO, "figs"))        # figure output
# run folder: first argument, else the RUN env var, else runs/seed42
R = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUN", os.path.join(REPO, "runs", "seed42"))
OUT = RESULTS; os.makedirs(OUT, exist_ok=True)
MODELS = ["resnet50", "vit_small", "effnet_b0_imnet", "zoobot", "effnet_b0_imnet_full", "zoobot_full"]
NAMES = {"resnet50": "ResNet-50", "vit_small": "ViT-S/16", "effnet_b0_imnet": "EffNet-B0 (ImageNet)",
         "zoobot": "Zoobot (EffNet-B0)", "effnet_b0_imnet_full": "EffNet-B0 ImageNet (full)", "zoobot_full": "Zoobot (full)"}
CLASSES = ["Disturbed", "Merging", "Round Smooth", "In-between Smooth", "Cigar Smooth", "Barred Spiral",
           "Unbarred Tight Spiral", "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge"]
# Galaxy10 SDSS class -> Galaxy10 DECaLS class (only morphologically equivalent classes)
S2D = {1: 2, 2: 3, 3: 4, 7: 6, 9: 7, 6: 8, 4: 9}
SHARED = sorted(S2D.values())

sp = np.load(glob.glob(f"{R}/**/split.npz", recursive=True)[0])
dlab, slab, itest, ival = sp["dlab"], sp["slab"], sp["test"], sp["val"]
yte, yva = dlab[itest], dlab[ival]
smask = np.isin(slab, list(S2D)); ys = np.array([S2D[c] for c in slab[smask]])
tmask = np.isin(yte, SHARED)


def softmax(z, T=1.0):
    z = z / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def ece(p, y, nb=15):
    conf, pred = p.max(1), p.argmax(1); acc = pred == y; e = 0
    edges = np.linspace(0, 1, nb + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def nll(p, y): return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()


def brier(p, y):
    o = np.zeros_like(p); o[np.arange(len(y)), y] = 1; return ((p - o) ** 2).sum(1).mean()


def fit_T(z, y):
    return minimize_scalar(lambda t: nll(softmax(z, t), y), bounds=(0.05, 10), method="bounded").x


def restrict(p):  # renormalise over the seven classes shared by both surveys
    q = np.zeros_like(p); q[:, SHARED] = p[:, SHARED]; return q / q.sum(1, keepdims=True)


def aurc(conf, correct):
    o = np.argsort(-conf); c = correct[o]
    risk = 1 - np.cumsum(c) / np.arange(1, len(c) + 1); return risk.mean()


def sel_acc(conf, correct, cov):
    o = np.argsort(-conf); k = max(1, int(round(cov * len(conf)))); return correct[o[:k]].mean()


def metrics(p, y, conf=None):
    pred = p.argmax(1); correct = pred == y
    conf = p.max(1) if conf is None else conf
    return dict(acc=100 * correct.mean(), f1=100 * f1_score(y, pred, average="macro"),
                bacc=100 * np.mean([correct[y == c].mean() for c in np.unique(y)]),
                ece=ece(p, y), nll=nll(p, y), brier=brier(p, y),
                auroc=roc_auc_score(correct, conf) if 0 < correct.mean() < 1 else np.nan,
                aurc=aurc(conf, correct), sel80=100 * sel_acc(conf, correct, 0.8))


def load(kind):
    runs = []
    for f in sorted(glob.glob(f"{R}/**/pred_{kind}_s*.npz", recursive=True)):
        d = dict(np.load(f)); s = int(f.split("_s")[-1].split(".")[0])
        d["res"] = json.load(open(f.replace("pred_", "res_").replace(".npz", ".json")))
        runs.append((s, d))
    return runs


rows, store = [], {}
for k in MODELS:
    runs = load(k); store[k] = runs
    if not runs: continue
    for s, d in runs:
        T = fit_T(d["logit_val"], yva)
        P = {n: softmax(d[f"logit_{n}"]) for n in ["test", "lowres", "sdss"]}
        PT = {n: softmax(d[f"logit_{n}"], T) for n in ["test", "lowres", "sdss"]}
        base = dict(model=k, seed=s, T=T, params=d["res"]["params_M"], lat=d["res"]["latency_ms"],
                    gpu=d["res"]["gpu"], train_min=d["res"]["train_min"])
        sets = {"decals10": (P["test"], yte, PT["test"]),
                "decals7": (restrict(P["test"][tmask]), yte[tmask], restrict(PT["test"][tmask])),
                "lowres7": (restrict(P["lowres"][tmask]), yte[tmask], restrict(PT["lowres"][tmask])),
                "sdss7": (restrict(P["sdss"][smask]), ys, restrict(PT["sdss"][smask])),
                "sdss7open": (P["sdss"][smask], ys, PT["sdss"][smask])}
        for nm, (p, y, pt) in sets.items():
            m = metrics(p, y); mt = metrics(pt, y)
            r = dict(base, set=nm, **m, ece_T=mt["ece"], nll_T=mt["nll"])
            # MC-dropout (mutual information) as the ranking score, same predictions as the MC mean
            src = {"decals10": "test", "decals7": "test", "lowres7": "lowres", "sdss7": "sdss", "sdss7open": "sdss"}[nm]
            msk = {"decals10": slice(None), "decals7": tmask, "lowres7": tmask, "sdss7": smask, "sdss7open": smask}[nm]
            pm = d[f"mc_{src}"][msk]; pm = restrict(pm) if nm in ("decals7", "lowres7", "sdss7") else pm
            mm = metrics(pm, y, conf=-d[f"unc_{src}"][msk][:, 1])
            r.update(mc_acc=mm["acc"], mc_ece=mm["ece"], mc_auroc=mm["auroc"], mc_aurc=mm["aurc"], mc_sel80=mm["sel80"])
            rows.append(r)
            # SDSS 20/80 recalibration with target labels (oracle-style reference)
            if nm == "sdss7":
                rng = np.random.default_rng(s); idx = rng.permutation(len(y)); cal, ev = idx[:len(y) // 5], idx[len(y) // 5:]
                z = d["logit_sdss"][smask]
                Ts = fit_T(np.log(np.clip(restrict(softmax(z))[cal], 1e-12, 1)), y[cal])
                pe = restrict(softmax(z))[ev]
                r["T_sdss"] = Ts; r["ece_sdss_before"] = ece(pe, y[ev])
                r["ece_sdss_after"] = ece(softmax(np.log(np.clip(pe, 1e-12, 1)), Ts), y[ev])

import pandas as pd
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/per_seed.csv", index=False)
agg = df.drop(columns=["gpu"]).groupby(["model", "set"]).agg(["mean", "std"]).round(4)
agg.to_csv(f"{OUT}/agg.csv")

# Deep ensembles (average of the seed members' probabilities)
ens_rows = []
for k in MODELS:
    runs = store[k]
    if len(runs) < 2: continue
    for nm, src, msk, yy, rs in [("decals10", "test", slice(None), yte, False), ("decals7", "test", tmask, yte[tmask], True),
                                 ("lowres7", "lowres", tmask, yte[tmask], True), ("sdss7", "sdss", smask, ys, True)]:
        Ps = np.stack([softmax(d[f"logit_{src}"])[msk] for _, d in runs])
        if rs: Ps = np.stack([restrict(p) for p in Ps])
        pm = Ps.mean(0)
        H = -(pm * np.log(np.clip(pm, 1e-12, 1))).sum(1)
        EH = -(Ps * np.log(np.clip(Ps, 1e-12, 1))).sum(2).mean(0)
        m = metrics(pm, yy, conf=-(H - EH)); m2 = metrics(pm, yy)
        ens_rows.append(dict(model=k, set=nm, members=len(runs), **{f"{a}": b for a, b in m2.items()},
                             mi_auroc=m["auroc"], mi_aurc=m["aurc"], mi_sel80=m["sel80"]))
pd.DataFrame(ens_rows).round(4).to_csv(f"{OUT}/ensemble.csv", index=False)

# Paired bootstrap for SDSS accuracy differences (first seed) and McNemar tests
first = {k: store[k][0][1] for k in MODELS if store.get(k)}
pred = {k: restrict(softmax(first[k]["logit_sdss"]))[smask].argmax(1) == ys for k in first}
rng = np.random.default_rng(0); B = 2000; bs = [rng.integers(0, len(ys), len(ys)) for _ in range(B)]
tests = []
for a, b in [("zoobot_full", "effnet_b0_imnet_full"), ("zoobot", "zoobot_full"), ("zoobot", "effnet_b0_imnet"), ("zoobot", "resnet50"), ("zoobot", "vit_small"),
             ("effnet_b0_imnet", "resnet50"), ("resnet50", "vit_small")]:
    if a not in pred or b not in pred: continue
    diff = np.array([pred[a][i].mean() - pred[b][i].mean() for i in bs]) * 100
    n01 = np.sum(pred[a] & ~pred[b]); n10 = np.sum(~pred[a] & pred[b])
    stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    tests.append(dict(a=a, b=b, diff=100 * (pred[a].mean() - pred[b].mean()), lo=np.percentile(diff, 2.5),
                      hi=np.percentile(diff, 97.5), n01=int(n01), n10=int(n10), p=chi2.sf(stat, 1)))
pd.DataFrame(tests).to_csv(f"{OUT}/paired_tests.csv", index=False)

# Per-class SDSS recall (mean over seeds) and SDSS confusion matrices for the first seed
pc = []
for k in MODELS:
    for s, d in store[k]:
        pr = restrict(softmax(d["logit_sdss"]))[smask].argmax(1)
        prt = restrict(softmax(d["logit_test"]))[tmask].argmax(1)
        for c in SHARED:
            pc.append(dict(model=k, seed=s, cls=CLASSES[c], sdss_recall=100 * (pr[ys == c] == c).mean(),
                           decals_recall=100 * (prt[yte[tmask] == c] == c).mean(), n_sdss=int((ys == c).sum()),
                           n_decals=int((yte[tmask] == c).sum())))
        pr10 = softmax(d["logit_test"]).argmax(1)
        for c in range(10):
            pc.append(dict(model=k, seed=s, cls=CLASSES[c], f1_10=100 * f1_score(yte == c, pr10 == c),
                           rec_10=100 * (pr10[yte == c] == c).mean()))
pd.DataFrame(pc).to_csv(f"{OUT}/per_class.csv", index=False)
for k in first:
    pr = restrict(softmax(first[k]["logit_sdss"]))[smask].argmax(1)
    np.save(f"{OUT}/cm_sdss_{k}.npy", confusion_matrix(ys, pr, labels=SHARED))
    np.save(f"{OUT}/cm_test_{k}.npy", confusion_matrix(yte, softmax(first[k]["logit_test"]).argmax(1), labels=range(10)))
print("SDSS shared-class counts", {CLASSES[c]: int((ys == c).sum()) for c in SHARED}, "N", len(ys))
print("DECaLS test shared", int(tmask.sum()), "test N", len(yte))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(df.groupby(["set", "model"])[["acc", "f1", "bacc", "ece", "ece_T", "nll", "auroc", "mc_auroc", "aurc", "sel80", "T"]].mean().round(3))
print(df.groupby(["set", "model"])[["acc", "f1", "ece"]].std().round(3))
print(pd.DataFrame(ens_rows).round(3))
print(pd.DataFrame(tests).round(4))
print(df[df.set == "sdss7"].groupby("model")[["T_sdss", "ece_sdss_before", "ece_sdss_after"]].mean().round(4))
print(df[df.set == "decals10"].groupby("model")[["params", "lat", "train_min"]].mean().round(2), df.gpu.unique())

# Bootstrap 95% intervals (first seed) for accuracy and ECE on each test condition
boot = []
for k in first:
    d = first[k]
    for nm, p, y in [("decals10", softmax(d["logit_test"]), yte),
                     ("decals7", restrict(softmax(d["logit_test"]))[tmask], yte[tmask]),
                     ("lowres7", restrict(softmax(d["logit_lowres"]))[tmask], yte[tmask]),
                     ("sdss7", restrict(softmax(d["logit_sdss"]))[smask], ys)]:
        rng = np.random.default_rng(1); A, E = [], []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y)); A.append(100 * (p[i].argmax(1) == y[i]).mean()); E.append(ece(p[i], y[i]))
        boot.append(dict(model=k, set=nm, acc_lo=np.percentile(A, 2.5), acc_hi=np.percentile(A, 97.5),
                         ece_lo=np.percentile(E, 2.5), ece_hi=np.percentile(E, 97.5)))
pd.DataFrame(boot).round(4).to_csv(f"{OUT}/bootstrap.csv", index=False)
print(pd.DataFrame(boot).round(3))
