"""Result figures from the saved predictions (single seed)."""
import glob, sys, numpy as np, pandas as pd, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyse_lib import *
import os
DATA = os.environ.get("DATA", "data")
FIGS = os.environ.get("FIGS", "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "legend.frameon": False})
COL = {"resnet50": "#4C72B0", "vit_small": "#8172B3", "effnet_b0_imnet_full": "#DD8452", "zoobot_full": "#55A868"}
SHORT = {"resnet50": "ResNet-50", "vit_small": "ViT-S/16", "effnet_b0_imnet_full": "EffNet-B0 (ImageNet)", "zoobot_full": "Zoobot"}
M = [k for k in ["resnet50", "vit_small", "effnet_b0_imnet_full", "zoobot_full"] if k in FIRST]


def save(name):
    plt.savefig(f"{FIGS}/{name}.pdf", bbox_inches="tight"); plt.savefig(f"{FIGS}/{name}.png", dpi=300, bbox_inches="tight"); plt.close()


# 1. accuracy on the three conditions (restricted, 7 classes) with bootstrap error bars
fig, ax = plt.subplots(figsize=(7.2, 3.0))
conds = [("decals7", "DECaLS test"), ("lowres7", "DECaLS at 69 px"), ("sdss7", "SDSS")]
w = 0.2
for j, k in enumerate(M):
    vals, lo, hi = [], [], []
    for c, _ in conds:
        p, y = probs(k, c); a = 100 * (p.argmax(1) == y).mean(); l, h = boot_acc(p, y); vals.append(a); lo.append(a - l); hi.append(h - a)
    x = np.arange(3) + (j - 1.5) * w
    ax.bar(x, vals, w * 0.92, color=COL[k], label=SHORT[k], yerr=[lo, hi], capsize=2, error_kw=dict(lw=0.8))
    for xi, v, e in zip(x, vals, hi):
        ax.text(xi, v + e + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(range(3)); ax.set_xticklabels([c[1] for c in conds]); ax.set_ylabel("Accuracy, 7 classes (%)")
ax.set_ylim(0, 112); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=8)
save("fig_accuracy")

# 2. reliability diagrams: DECaLS test (10 classes) and SDSS (7 classes), before/after source temperature
fig, axes = plt.subplots(2, len(M), figsize=(7.2, 3.8), sharex=True, sharey=True)
for j, k in enumerate(M):
    for i, (c, lab) in enumerate([("decals10", "DECaLS"), ("sdss7", "SDSS")]):
        ax = axes[i, j]; p, y = probs(k, c)
        edges = np.linspace(0, 1, 16); conf = p.max(1); acc = p.argmax(1) == y
        cen, accs, cnt = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() > 0: cen.append(conf[m].mean()); accs.append(acc[m].mean()); cnt.append(m.sum())
        ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=0.8)
        ax.bar(edges[:-1] + 1 / 30, [0] * 15, 1 / 15)
        for cc, aa in zip(cen, accs):
            ax.bar(cc, aa, 1 / 15 * 0.9, color=COL[k], alpha=0.85, edgecolor="white", lw=0.4)
        ax.text(0.04, 0.9, f"ECE {ece(p, y):.3f}", fontsize=7.5, transform=ax.transAxes)
        if i == 0: ax.set_title(SHORT[k], fontsize=8.5)
        if j == 0: ax.set_ylabel(f"{lab}\naccuracy")
        if i == 1: ax.set_xlabel("confidence")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save("fig_reliability")

# 3. risk-coverage on DECaLS and SDSS (softmax confidence)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
for ax, (c, t) in zip(axes, [("decals10", "DECaLS test, 10 classes"), ("sdss7", "SDSS, 7 classes")]):
    for k in M:
        p, y = probs(k, c); o = np.argsort(-p.max(1)); corr = (p.argmax(1) == y)[o]
        cov = np.arange(1, len(y) + 1) / len(y); acc = np.cumsum(corr) / np.arange(1, len(y) + 1)
        ax.plot(cov, 100 * acc, color=COL[k], lw=1.4, label=SHORT[k])
    ax.axvline(0.8, color="#999999", lw=0.7, ls=":")
    ax.set_xlabel("coverage"); ax.set_ylabel("selective accuracy (%)"); ax.set_title(t, fontsize=9); ax.set_xlim(0.05, 1)
axes[0].legend(fontsize=7.5, loc="lower left")
save("fig_selective")

# 4. SDSS confusion matrices (row-normalised), ResNet-50 vs Zoobot
names = [CLASSES[c] for c in SHARED]
short = ["Round sm.", "In-betw. sm.", "Cigar sm.", "Tight sp.", "Loose sp.", "Edge-on no b.", "Edge-on bulge"]
sel = [k for k in ["resnet50", "zoobot_full"] if k in M]
fig, axes = plt.subplots(1, len(sel), figsize=(7.4, 3.6), gridspec_kw=dict(wspace=0.08))
for ax, k in zip(np.atleast_1d(axes), sel):
    cm = np.load(f"results/cm_sdss_{k}.npy").astype(float); cm = 100 * cm / cm.sum(1, keepdims=True)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    for a in range(7):
        for b in range(7):
            ax.text(b, a, f"{cm[a, b]:.0f}", ha="center", va="center", fontsize=6.5, color="white" if cm[a, b] > 55 else "#222222")
    ax.set_xticks(range(7)); ax.set_yticks(range(7)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7); ax.set_title(f"{SHORT[k]} on SDSS", fontsize=9); ax.set_xlabel("predicted")
    for s in ax.spines.values(): s.set_visible(False)
np.atleast_1d(axes)[0].set_ylabel("true")
for a in np.atleast_1d(axes)[1:]: a.set_yticklabels([])
save("fig_cm_sdss")

# 5. DECaLS 10-class confusion matrix of Zoobot and ResNet-50
cls_short = ["Disturbed", "Merging", "Round sm.", "In-betw. sm.", "Cigar sm.", "Barred sp.", "Tight sp.", "Loose sp.", "Edge-on no b.", "Edge-on bulge"]
fig, axes = plt.subplots(1, len(sel), figsize=(7.6, 3.8), gridspec_kw=dict(wspace=0.08))
for ax, k in zip(np.atleast_1d(axes), sel):
    cm = np.load(f"results/cm_test_{k}.npy").astype(float); cm = 100 * cm / cm.sum(1, keepdims=True)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    for a in range(10):
        for b in range(10):
            if cm[a, b] >= 0.5:
                ax.text(b, a, f"{cm[a, b]:.0f}", ha="center", va="center", fontsize=5.5, color="white" if cm[a, b] > 55 else "#222222")
    ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.set_xticklabels(cls_short, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticklabels(cls_short, fontsize=6.5); ax.set_title(f"{SHORT[k]}, DECaLS test", fontsize=9); ax.set_xlabel("predicted")
    for s in ax.spines.values(): s.set_visible(False)
np.atleast_1d(axes)[0].set_ylabel("true")
for a in np.atleast_1d(axes)[1:]: a.set_yticklabels([])
save("fig_cm_decals")

# 6. Grad-CAM examples: same class, DECaLS vs SDSS, all models
import torch, torch.nn.functional as F
def r224(a):
    x = torch.from_numpy(np.asarray(a)[None]).permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)[0].permute(1, 2, 0).numpy() / 255
k0 = M[0]; d0 = FIRST[k0]
with h5py.File(f"{DATA}/Galaxy10_DECals.h5") as fd, h5py.File(f"{DATA}/Galaxy10.h5") as fs:
    picks = []
    for c in [2, 6, 9]:  # round smooth, tight spiral, edge-on with bulge
        tp = [i for i, t in enumerate(d0["cam_test_pos"]) if YTE[t] == c][0]
        sc = [s for s, (sid) in enumerate(d0["cam_sdss_pos"]) if SLAB[sid] in S2D and S2D[SLAB[sid]] == c][0]
        picks.append((c, tp, sc))
    fig, axes = plt.subplots(len(picks) * 2, len(M) + 1, figsize=(6.2, 7.2))
    for r, (c, tp, sc) in enumerate(picks):
        for q, (src, pos, key, lab) in enumerate([(fd, ITEST[d0["cam_test_pos"][tp]], "cam_test", "DECaLS"),
                                                  (fs, d0["cam_sdss_pos"][sc], "cam_sdss", "SDSS")]):
            img = r224(src["images"][int(pos)]); row = 2 * r + q
            axes[row, 0].imshow(img); axes[row, 0].set_ylabel(f"{lab}\n{CLASSES[c].replace('Unbarred ', '')}", fontsize=8.5)
            for j, k in enumerate(M):
                cam = FIRST[k][key][tp if q == 0 else sc].astype(float)
                axes[row, j + 1].imshow(img); axes[row, j + 1].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                if row == 0: axes[row, j + 1].set_title(SHORT[k], fontsize=9)
            if row == 0: axes[row, 0].set_title("input", fontsize=9)
    for a in axes.ravel(): a.set_xticks([]); a.set_yticks([])
    for s in axes.ravel():
        for sp in s.spines.values(): sp.set_visible(False)
    plt.subplots_adjust(wspace=0.03, hspace=0.06)
    save("fig_gradcam")

# 7. training curves
import json
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for k in M:
    h = FIRST[k]["res"]["hist"]
    axes[0].plot(range(1, len(h) + 1), [e["val_loss"] for e in h], color=COL[k], lw=1.2, label=SHORT[k])
    axes[1].plot(range(1, len(h) + 1), [100 * e["val_acc"] for e in h], color=COL[k], lw=1.2)
axes[0].set_ylabel("validation NLL"); axes[1].set_ylabel("validation accuracy (%)")
for a in axes: a.set_xlabel("epoch")
axes[0].legend(fontsize=7.5)
save("fig_curves")
print("figures done")

# 8. example images: one galaxy per shared class in DECaLS and SDSS
with h5py.File(f"{DATA}/Galaxy10_DECals.h5") as fd, h5py.File(f"{DATA}/Galaxy10.h5") as fs:
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(2, 7, figsize=(7.4, 2.5))
    for j, c in enumerate(SHARED):
        i_d = int(rng.choice(ITEST[YTE == c])); sc = [s for s, d_ in S2D.items() if d_ == c][0]
        i_s = int(rng.choice(np.where(SLAB == sc)[0]))
        axes[0, j].imshow(np.asarray(fd["images"][i_d])); axes[1, j].imshow(np.asarray(fs["images"][i_s]))
        axes[0, j].set_title(CLASSES[c].replace("Unbarred ", "").replace(" without", " no").replace("Edge-on with Bulge", "Edge-on bulge"), fontsize=7)
    axes[0, 0].set_ylabel("DECaLS\n256 px", fontsize=8); axes[1, 0].set_ylabel("SDSS\n69 px", fontsize=8)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
    plt.subplots_adjust(wspace=0.04, hspace=0.12)
    save("fig_examples")

# 9. confidence of correct and wrong predictions, DECaLS vs SDSS
fig, axes = plt.subplots(2, len(M), figsize=(7.2, 3.6), sharex=True)
for j, k in enumerate(M):
    for i, (c, lab) in enumerate([("decals10", "DECaLS"), ("sdss7", "SDSS")]):
        p, y = probs(k, c); conf = p.max(1); ok = p.argmax(1) == y; ax = axes[i, j]
        bins = np.linspace(0, 1, 26)
        ax.hist(conf[ok], bins, density=True, color=COL[k], alpha=0.75, label="correct")
        ax.hist(conf[~ok], bins, density=True, histtype="step", color="#222222", lw=1.1, label="wrong")
        ax.set_yticks([])
        if i == 0: ax.set_title(SHORT[k], fontsize=8.5)
        if j == 0: ax.set_ylabel(f"{lab}\ndensity")
        if i == 1: ax.set_xlabel("max. softmax prob.")
axes[0, 0].legend(fontsize=7, loc="upper left")
save("fig_confidence")
print("extra figures done")

# 10. DECaLS sample grid, two images per class
with h5py.File(f"{DATA}/Galaxy10_DECals.h5") as fd:
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 10, figsize=(7.4, 1.95))
    names = ["Disturbed", "Merging", "Round\nsmooth", "In-between\nsmooth", "Cigar\nsmooth", "Barred\nspiral",
             "Tight\nspiral", "Loose\nspiral", "Edge-on\nno bulge", "Edge-on\nwith bulge"]
    for c in range(10):
        for r_, i in enumerate(sorted(rng.choice(np.where(DLAB == c)[0], 2, replace=False))):
            axes[r_, c].imshow(np.asarray(fd["images"][int(i)]))
        axes[0, c].set_title(names[c], fontsize=7.5)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    save("fig_samples")

# 11. class distribution of both datasets
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[10, 10]))
cnt = np.bincount(DLAB, minlength=10)
axes[0].barh(range(10), cnt, color="#6c8ebf"); axes[0].set_yticks(range(10)); axes[0].set_yticklabels([n.replace("\n", " ") for n in names], fontsize=8)
axes[0].invert_yaxis(); axes[0].set_xlabel("images"); axes[0].set_title("Galaxy10 DECaLS (17,736)", fontsize=9)
for i, v in enumerate(cnt): axes[0].text(v + 30, i, f"{v:,}", va="center", fontsize=7)
sn = ["Face-on, no spiral", "Smooth, round", "Smooth, in-between", "Smooth, cigar", "Edge-on, rounded bulge",
      "Edge-on, boxy bulge", "Edge-on, no bulge", "Face-on, tight spiral", "Face-on, medium spiral", "Face-on, loose spiral"]
cs = np.bincount(SLAB, minlength=10); colr = ["#82b366" if k in S2D else "#bbbbbb" for k in range(10)]
axes[1].barh(range(10), cs, color=colr); axes[1].set_yticks(range(10)); axes[1].set_yticklabels(sn, fontsize=8)
axes[1].invert_yaxis(); axes[1].set_xlabel("images"); axes[1].set_title("Galaxy10 SDSS (21,785)", fontsize=9)
for i, v in enumerate(cs): axes[1].text(v + 60, i, f"{v:,}", va="center", fontsize=7)
axes[0].set_xlim(0, cnt.max() * 1.2); axes[1].set_xlim(0, cs.max() * 1.22)
plt.tight_layout()
save("fig_classdist")

# 12. per-class recall, DECaLS vs SDSS (shared classes)
lab7 = ["Round sm.", "In-betw. sm.", "Cigar sm.", "Tight sp.", "Loose sp.", "Edge-on no b.", "Edge-on bulge"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
for ax, (cond, t) in zip(axes, [("decals7", "DECaLS test"), ("sdss7", "SDSS")]):
    for j, k in enumerate(M):
        p, y = probs(k, cond); pr = p.argmax(1)
        rec = [100 * (pr[y == c] == c).mean() for c in SHARED]
        ax.bar(np.arange(7) + (j - 1.5) * 0.2, rec, 0.19, color=COL[k], label=SHORT[k])
    ax.set_xticks(range(7)); ax.set_xticklabels(lab7, fontsize=7.5, rotation=40, ha="right"); ax.set_title(t, fontsize=9)
    ax.set_ylim(0, 105)
axes[0].set_ylabel("recall (%)"); h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
fig.subplots_adjust(top=0.82, bottom=0.27, wspace=0.06)
save("fig_recall")
print("more figures done")
