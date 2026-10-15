"""Mass-in-galaxy score for Grad-CAM maps (Eq. mig)."""
import glob, os, sys, numpy as np, h5py, pandas as pd, torch, torch.nn.functional as F
from scipy import ndimage
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DATA", os.path.join(REPO, "data"))        # folder with the .h5 files
RESULTS = os.environ.get("RESULTS", os.path.join(REPO, "results"))  # csv / npy output
FIGS = os.environ.get("FIGS", os.path.join(REPO, "figs"))        # figure output
R = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUN", os.path.join(REPO, "runs", "seed42"))
os.makedirs(RESULTS, exist_ok=True)
sp = np.load(glob.glob(f"{R}/**/split.npz", recursive=True)[0]); itest = sp["test"]


def to224(a):
    x = torch.from_numpy(a).permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1).numpy()


def mask(img):
    L = img @ np.array([0.299, 0.587, 0.114])
    med = np.median(L); mad = np.median(np.abs(L - med)) * 1.4826
    m = L > med + 3 * mad
    lab, n = ndimage.label(m)
    if n == 0: return m
    cy, cx = np.array(L.shape) / 2
    best, bd = 0, 1e9
    for j in range(1, n + 1):
        yy, xx = np.nonzero(lab == j); dd = np.min((yy - cy) ** 2 + (xx - cx) ** 2)
        if dd < bd: bd, best = dd, j
    return lab == best


rows = []
with h5py.File(f"{DATA}/Galaxy10_DECals.h5") as fd, h5py.File(f"{DATA}/Galaxy10.h5") as fs:
    for f in sorted(glob.glob(f"{R}/**/pred_*_s42.npz", recursive=True)):
        k = f.split("pred_")[1].split("_s42")[0]; d = np.load(f)
        if "cam_test" not in d: continue
        for nm, pos, cams, src in [("decals", itest[d["cam_test_pos"]], d["cam_test"], fd), ("sdss", d["cam_sdss_pos"], d["cam_sdss"], fs)]:
            o = np.argsort(pos); imgs = to224(src["images"][np.sort(pos)])
            for j, i in enumerate(o):
                g = mask(imgs[j]); h = cams[i].astype(np.float64)
                rows.append(dict(model=k, set=nm, mig=(h * g).sum() / max(h.sum(), 1e-9), area=g.mean()))
df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, "mig.csv"), index=False)
print(df.groupby(["model", "set"]).agg(["mean", "std"]).round(3))
