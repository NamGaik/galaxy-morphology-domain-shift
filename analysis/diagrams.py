"""Block diagrams in a flat flow-chart style (rounded boxes, orthogonal connectors)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGS = os.environ.get("FIGS", "figs")
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5})
N_SDSS = os.environ.get("N_SDSS", "17,401")

BLUE = ("#dae8fc", "#6c8ebf"); GREEN = ("#d5e8d4", "#82b366"); ORANGE = ("#ffe6cc", "#d79b00")
PURPLE = ("#e1d5e7", "#9673a6"); GREY = ("#f5f5f5", "#666666"); YEL = ("#fff2cc", "#d6b656")
LINE = "#333333"
FS = 8.5


def box(ax, x, y, w, h, text, c, fs=FS, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=c[0], ec=c[1], lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.1, color=LINE, shrinkA=0, shrinkB=0))


def poly(ax, pts):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=LINE, lw=1.1, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1])


def group(ax, x, y, w, h, title, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15", fc="none", ec=c[1], lw=1.0,
                                ls=(0, (5, 3))))
    ax.text(x + 0.15, y + h - 0.12, title, ha="left", va="top", fontsize=FS, fontweight="bold", color=c[1])


def save(fig, name):
    fig.savefig(f"{FIGS}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{FIGS}/{name}.png", dpi=300, bbox_inches="tight")


# ---------------- pipeline ----------------
fig, ax = plt.subplots(figsize=(7.4, 5.0)); ax.set_xlim(0, 15.2); ax.set_ylim(0, 10.2); ax.axis("off")
group(ax, 0.1, 4.75, 3.7, 5.35, "Data", GREY)
box(ax, 0.3, 7.2, 3.3, 2.1, "Galaxy10 DECaLS\n17,736 images\n256 x 256 px\n10 classes", BLUE)
box(ax, 0.3, 4.95, 3.3, 2.1, "Galaxy10 SDSS\n21,785 images\n69 x 69 px\n10 classes", BLUE)
box(ax, 4.4, 7.2, 3.0, 2.1, "Stratified split\n(seed 42)\n12,414 / 2,661\n/ 2,661", GREY)
box(ax, 4.4, 4.95, 3.0, 2.1, f"Class matching\n7 shared\nclasses\nN = {N_SDSS}", GREY)
box(ax, 4.4, 2.6, 3.0, 2.0, "Degraded\nDECaLS test\n256 -> 69 ->\n224 px", YEL)
arrow(ax, (3.6, 8.25), (4.4, 8.25)); arrow(ax, (3.6, 6.0), (4.4, 6.0))
poly(ax, [(7.4, 7.6), (7.75, 7.6), (7.75, 3.6), (7.4, 3.6)])
group(ax, 8.1, 4.75, 3.4, 5.35, "Training", ORANGE)
box(ax, 8.3, 8.1, 3.0, 1.2, "ResNet-50\n(ImageNet)", ORANGE)
box(ax, 8.3, 6.7, 3.0, 1.2, "ViT-S/16\n(ImageNet)", ORANGE)
box(ax, 8.3, 4.95, 3.0, 1.55, "EfficientNet-B0\n(ImageNet or\nZoobot weights)", ORANGE)
poly(ax, [(7.4, 8.7), (8.3, 8.7)])
box(ax, 0.3, 0.3, 10.9, 1.8, "Inference with no target-survey training or adaptation\n"
    "(i) DECaLS test   (ii) degraded DECaLS test\n(iii) SDSS (resolution + survey change)", PURPLE)
poly(ax, [(9.8, 4.95), (9.8, 2.1)]); poly(ax, [(5.9, 2.6), (5.9, 2.1)])
poly(ax, [(4.4, 5.3), (4.05, 5.3), (4.05, 2.1)])
group(ax, 11.9, 0.1, 3.2, 10.0, "Evaluation", GREEN)
labels = ["Accuracy,\nmacro-F1,\nbal. accuracy", "ECE, NLL, Brier,\ntemperature\nscaling", "MC dropout\n(mutual\ninformation)",
          "Selective\nprediction\n(AUROC, AURC)", "Grad-CAM,\nmass-in-galaxy\nscore", "Bootstrap CI,\nMcNemar test"]
for i, t in enumerate(labels):
    box(ax, 12.1, 8.15 - i * 1.6, 2.8, 1.4, t, GREEN)
poly(ax, [(11.2, 1.2), (11.9, 1.2)])
save(fig, "fig_pipeline")

# ---------------- model families ----------------
fig, ax = plt.subplots(figsize=(7.4, 3.9)); ax.set_xlim(0, 14.2); ax.set_ylim(0, 7.0); ax.axis("off")
rows = [("ResNet-50", "residual CNN\nImageNet-1k", "2048-d", "Dropout(0.2)\nLinear(10)", BLUE, "23.5 M"),
        ("ViT-S/16", "transformer\n196 patches + [CLS]", "384-d", "Dropout(0.2)\nLinear(10)", PURPLE, "21.7 M"),
        ("EfficientNet-B0", "MBConv CNN\nImageNet-1k", "1280-d", "Linear(256), ReLU\nDropout(0.2)\nLinear(10)", ORANGE, "4.3 M"),
        ("Zoobot", "EfficientNet-B0\nGalaxy Zoo weights", "1280-d", "Linear(256), ReLU\nDropout(0.2)\nLinear(10)", GREEN, "4.3 M")]
for i, (n, e, d, h, c, p) in enumerate(rows):
    y = 5.15 - i * 1.7
    box(ax, 0.1, y, 1.9, 1.4, "image\n224 x 224\nx 3", GREY)
    box(ax, 2.6, y, 3.7, 1.4, f"{n}\n{e}", c)
    box(ax, 6.9, y, 1.6, 1.4, f"pooled\nfeature\n{d}", GREY)
    box(ax, 9.1, y, 2.8, 1.4, h, c)
    box(ax, 12.5, y, 1.65, 1.4, f"softmax\n10 classes\n({p})", GREY)
    for a, b in [(2.0, 2.6), (6.3, 6.9), (8.5, 9.1), (11.9, 12.5)]:
        arrow(ax, (a, y + 0.7), (b, y + 0.7))
ax.text(4.45, 6.75, "encoder (fine-tuned)", ha="center", fontsize=FS, fontweight="bold")
ax.text(10.5, 6.75, "task head (new)", ha="center", fontsize=FS, fontweight="bold")
save(fig, "fig_models")
print("diagrams written")
