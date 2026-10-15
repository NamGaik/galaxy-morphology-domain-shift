"""Replace figure and table captions in the paper sources by short ones, keyed by label."""
import glob, re

CAP = {
    "fig:pipeline": "Outline of the experiments",
    "fig:models": "The four classifiers; the two EfficientNet-B0 models differ only in their initial weights",
    "fig:samples": "Example Galaxy10 DECaLS images, two per class",
    "fig:classdist": "Class sizes of the two datasets; grey SDSS classes have no DECaLS counterpart",
    "fig:examples": "One galaxy per shared class in DECaLS (top) and SDSS (bottom)",
    "fig:curves": "Validation NLL and accuracy per epoch (recipe A). The missing ResNet-50 point at epoch 4 had a non-finite half-precision loss",
    "fig:cmdecals": "Row-normalised confusion matrices (\\%) on the DECaLS test split",
    "fig:acc": "Seven-class accuracy under the three test conditions, with 95\\,\\% bootstrap intervals",
    "fig:recall": "Per-class recall on the shared classes, DECaLS test (left) and SDSS (right)",
    "fig:cmsdss": "Row-normalised confusion matrices (\\%) on SDSS",
    "fig:rel": "Reliability diagrams on DECaLS (top) and SDSS (bottom) before temperature scaling",
    "fig:conf": "Maximum softmax probability of correct and wrong predictions",
    "fig:sel": "Selective accuracy against coverage; the dotted line marks 80\\,\\% coverage",
    "fig:gradcam": "Grad-CAM maps for the predicted class, DECaLS and SDSS examples",
    "tab:map": "Class correspondence between Galaxy10 SDSS and Galaxy10 DECaLS",
    "tab:data": "Galaxy10 DECaLS images per class and split",
    "tab:decals": "Results on the DECaLS test split (ten classes). A, B: training recipe; IN: ImageNet weights",
    "tab:cost": "Model size, single-image latency on a Tesla T4 and training time",
    "tab:f1": "Per-class F1 (\\%) on the DECaLS test split; best value in bold",
    "tab:shift": "Seven-class accuracy (\\%) under the three test conditions and its decomposition (Eq.~(\\ref{eq:decomp}))",
    "tab:perclass": "Per-class recall (\\%), DECaLS test / SDSS",
    "tab:tests": "Paired comparisons on SDSS (bootstrap 95\\,\\% CI and McNemar test)",
    "tab:calib": "Calibration before and after source and target temperature scaling",
    "tab:unc": "Error detection and selective prediction",
    "tab:mig": "Share of Grad-CAM mass inside the galaxy mask (MiG) and mask area",
}


def fix(s):
    out, i = [], 0
    while True:
        j = s.find("\\caption{", i)
        if j < 0:
            out.append(s[i:]); break
        k, depth = j + len("\\caption{"), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(s[k], 0); k += 1
        m = re.search(r"\\label\{([^}]*)\}", s[k:k + 200])
        lab = m.group(1) if m else None
        out.append(s[i:j]); out.append("\\caption{" + CAP[lab] + "}" if lab in CAP else s[j:k]); i = k
    return "".join(out)


if __name__ == "__main__":
    import sys
    for f in glob.glob(sys.argv[1] + "/*.tex"):
        t = open(f, encoding="ascii").read(); n = fix(t)
        if n != t:
            open(f, "w", encoding="ascii").write(n); print("captions:", f)
