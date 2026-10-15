"""Shared helpers for the figure script."""
import glob, json, os
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DATA", os.path.join(REPO, "data"))        # folder with the .h5 files
RESULTS = os.environ.get("RESULTS", os.path.join(REPO, "results"))  # csv / npy output
FIGS = os.environ.get("FIGS", os.path.join(REPO, "figs"))        # figure output
# run folder with split.npz, pred_*.npz and res_*.json (set with the RUN env var)
R = os.environ.get("RUN", os.path.join(REPO, "runs", "seed42"))
MODELS = ["resnet50", "vit_small", "effnet_b0_imnet", "zoobot", "effnet_b0_imnet_full", "zoobot_full"]
CLASSES = ["Disturbed", "Merging", "Round Smooth", "In-between Smooth", "Cigar Smooth", "Barred Spiral",
           "Unbarred Tight Spiral", "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge"]
S2D = {1: 2, 2: 3, 3: 4, 7: 6, 9: 7, 6: 8, 4: 9}
SHARED = sorted(S2D.values())
sp = np.load(glob.glob(f"{R}/**/split.npz", recursive=True)[0])
DLAB, SLAB, ITEST = sp["dlab"], sp["slab"], sp["test"]
YTE = DLAB[ITEST]
SMASK = np.isin(SLAB, list(S2D)); YS = np.array([S2D[c] for c in SLAB[SMASK]]); TMASK = np.isin(YTE, SHARED)
FIRST = {}
for k in MODELS:
    f = glob.glob(f"{R}/**/pred_{k}_s42.npz", recursive=True)
    if f:
        FIRST[k] = dict(np.load(f[0])); FIRST[k]["res"] = json.load(open(f[0].replace("pred_", "res_").replace(".npz", ".json")))


def softmax(z, T=1.0):
    z = z / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def restrict(p):
    q = np.zeros_like(p); q[:, SHARED] = p[:, SHARED]; return q / q.sum(1, keepdims=True)


def ece(p, y, nb=15):
    conf, acc = p.max(1), p.argmax(1) == y; e = 0; edges = np.linspace(0, 1, nb + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def probs(k, s):
    d = FIRST[k]
    if s == "decals10": return softmax(d["logit_test"]), YTE
    if s == "decals7": return restrict(softmax(d["logit_test"]))[TMASK], YTE[TMASK]
    if s == "lowres7": return restrict(softmax(d["logit_lowres"]))[TMASK], YTE[TMASK]
    if s == "sdss7": return restrict(softmax(d["logit_sdss"]))[SMASK], YS


def boot_acc(p, y, B=2000):
    rng = np.random.default_rng(1); c = p.argmax(1) == y
    a = [100 * c[rng.integers(0, len(y), len(y))].mean() for _ in range(B)]
    return np.percentile(a, 2.5), np.percentile(a, 97.5)
