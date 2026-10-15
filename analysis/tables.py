"""Write LaTeX tables for the paper from results/*.csv."""
import os
import numpy as np, pandas as pd
P = os.environ.get("TEX_OUT", "tex/")
os.makedirs(P, exist_ok=True)
r = pd.read_csv("results/per_seed.csv"); b = pd.read_csv("results/bootstrap.csv")
pc = pd.read_csv("results/per_class.csv"); t = pd.read_csv("results/paired_tests.csv")
NM = {"resnet50": "ResNet-50", "vit_small": "ViT-S/16", "effnet_b0_imnet": "EffNet-B0 IN (B)",
      "zoobot": "Zoobot (B)", "effnet_b0_imnet_full": "EffNet-B0 IN (A)", "zoobot_full": "Zoobot (A)"}
ORDER = [m for m in ["resnet50", "vit_small", "effnet_b0_imnet_full", "zoobot_full", "effnet_b0_imnet", "zoobot"] if m in set(r.model)]


def g(m, s, c): return r[(r.model == m) & (r.set == s)][c].iloc[0]


def ci(m, s, c): x = b[(b.model == m) & (b.set == s)].iloc[0]; return x[c + "_lo"], x[c + "_hi"]


def bold_best(vals, hi=True):
    k = np.nanargmax(vals) if hi else np.nanargmin(vals); return k


def write(name, body):
    # scale wide tables to the text width
    body = body.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}") \
        .replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}")
    for a, b in [(" -", " $-$"), ("[-", "[$-$"), (", -", ", $-$")]:
        body = body.replace(a + "0", b + "0").replace(a + "1", b + "1").replace(a + "2", b + "2").replace(a + "3", b + "3") \
            .replace(a + "4", b + "4").replace(a + "5", b + "5").replace(a + "6", b + "6").replace(a + "7", b + "7") \
            .replace(a + "8", b + "8").replace(a + "9", b + "9")
    from captions import fix
    open(P + name, "w", encoding="ascii").write(fix(body))


# Table: DECaLS test, 10 classes
rows = []
acc = [g(m, "decals10", "acc") for m in ORDER]; kb = bold_best(acc)
for i, m in enumerate(ORDER):
    lo, hi = ci(m, "decals10", "acc"); a = f"{g(m,'decals10','acc'):.2f}"
    a = f"\\textbf{{{a}}}" if i == kb else a
    rows.append(f"{NM[m]} & {a} [{lo:.1f}, {hi:.1f}] & {g(m,'decals10','f1'):.2f} & {g(m,'decals10','bacc'):.2f} & "
                f"{g(m,'decals10','ece'):.3f} & {g(m,'decals10','nll'):.3f} & {g(m,'decals10','brier'):.3f} \\\\")
write("tab_decals.tex", r"""\begin{table*}[t]
\caption{In-survey results on the Galaxy10 DECaLS test split (2{,}661 images, ten classes). Accuracy is given with its 95\,\% bootstrap interval. ECE, NLL and Brier score are before temperature scaling. Recipe A: 30 epochs of full fine-tuning at learning rate $10^{-4}$; recipe B: 10 frozen-encoder epochs at $10^{-3}$ followed by 20 full epochs at $10^{-5}$. IN: ImageNet weights}
\label{tab:decals}
\centering
\begin{tabular}{@{}lcccccc@{}}
\toprule
Model & Acc. (\%) [95\,\% CI] & Macro-F1 & Bal. acc. & ECE & NLL & Brier\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
rows = [f"{NM[m]} & {g(m,'decals10','params'):.2f} & {g(m,'decals10','lat'):.2f} & {g(m,'decals10','train_min'):.0f} \\\\" for m in ORDER]
write("tab_cost.tex", r"""\begin{table*}[t]
\caption{Model size and cost. Latency is the mean time for one image (batch size one, FP32) on a Tesla T4; training time is wall-clock minutes on the same GPU with mixed precision}
\label{tab:cost}
\centering
\begin{tabular}{@{}lccc@{}}
\toprule
Model & Parameters (M) & Latency (ms) & Training (min)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# Table: three conditions, 7 classes restricted
rows = []
for m in ORDER:
    a7, al, asd = g(m, "decals7", "acc"), g(m, "lowres7", "acc"), g(m, "sdss7", "acc")
    lo, hi = ci(m, "sdss7", "acc")
    rows.append(f"{NM[m]} & {a7:.2f} & {al:.2f} & {asd:.2f} [{lo:.1f}, {hi:.1f}] & {g(m,'sdss7open','acc'):.2f} & "
                f"{al-a7:+.2f} & {asd-al:+.2f} & {asd-a7:+.2f} & {g(m,'sdss7','f1'):.2f} & {g(m,'sdss7','bacc'):.2f} \\\\")
write("tab_shift.tex", r"""\begin{table*}[t]
\caption{Accuracy (\%) on the seven shared classes under the three test conditions, and the decomposition of Eq.~(\ref{eq:decomp}). DECaLS and DECaLS at 69\,px use the 1{,}914 test images of the shared classes; SDSS uses 17{,}401 images. All values use the restricted protocol of Eq.~(\ref{eq:restrict}) except the column ``SDSS open'', where the model keeps all ten outputs. Macro-F1 and balanced accuracy are for SDSS}
\label{tab:shift}
\centering
\begin{tabular}{@{}lccccccccc@{}}
\toprule
 & \multicolumn{4}{c}{Accuracy} & \multicolumn{3}{c}{Change (points)} & \multicolumn{2}{c}{SDSS}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(l){9-10}
Model & DECaLS & 69\,px & SDSS [95\,\% CI] & SDSS open & $\Delta_{\mathrm{res}}$ & $\Delta_{\mathrm{survey}}$ & $\Delta_{\mathrm{total}}$ & Macro-F1 & Bal. acc.\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# Table: calibration
rows = []
for m in ORDER:
    rows.append(f"{NM[m]} & {g(m,'decals10','T'):.2f} & {g(m,'decals10','ece'):.3f} & {g(m,'decals10','ece_T'):.3f} & "
                f"{g(m,'sdss7','ece'):.3f} & {g(m,'sdss7','ece_T'):.3f} & {g(m,'sdss7','nll'):.2f} & {g(m,'sdss7','nll_T'):.2f} & "
                f"{g(m,'sdss7','T_sdss'):.2f} & {g(m,'sdss7','ece_sdss_before'):.3f} & {g(m,'sdss7','ece_sdss_after'):.3f} \\\\")
write("tab_calib.tex", r"""\begin{table*}[t]
\caption{Calibration before and after temperature scaling. Source scaling fits $T_{\mathrm{src}}$ on the DECaLS validation split and applies it everywhere; target scaling fits $T_{\mathrm{tgt}}$ on 20\,\% of the matched SDSS images and is evaluated on the other 80\,\% (13{,}921 images). ECE uses 15 bins}
\label{tab:calib}
\centering
\begin{tabular}{@{}lcccccccccc@{}}
\toprule
 & \multicolumn{3}{c}{DECaLS test (10 classes)} & \multicolumn{4}{c}{SDSS, source scaling} & \multicolumn{3}{c}{SDSS, target scaling}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-8}\cmidrule(l){9-11}
Model & $T_{\mathrm{src}}$ & ECE & ECE$_T$ & ECE & ECE$_T$ & NLL & NLL$_T$ & $T_{\mathrm{tgt}}$ & ECE & ECE$_T$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# Table: uncertainty and selective prediction
rows = []
for m in ORDER:
    rows.append(f"{NM[m]} & {g(m,'decals10','auroc'):.3f} & {g(m,'decals10','mc_auroc'):.3f} & {g(m,'decals10','aurc'):.3f} & "
                f"{g(m,'decals10','sel80'):.2f} & {g(m,'sdss7','auroc'):.3f} & {g(m,'sdss7','mc_auroc'):.3f} & {g(m,'sdss7','aurc'):.3f} & "
                f"{g(m,'sdss7','sel80'):.2f} \\\\")
write("tab_unc.tex", r"""\begin{table*}[t]
\caption{Error detection and selective prediction. AUROC measures how well a score separates correct from wrong predictions (0.5 is chance). MSP is the maximum softmax probability and MI the mutual information from 30 Monte Carlo dropout passes. AURC (lower is better) and accuracy at 80\,\% coverage use MSP}
\label{tab:unc}
\centering
\begin{tabular}{@{}lcccccccc@{}}
\toprule
 & \multicolumn{4}{c}{DECaLS test (10 classes)} & \multicolumn{4}{c}{SDSS (7 classes)}\\
\cmidrule(lr){2-5}\cmidrule(l){6-9}
Model & AUROC$_{\mathrm{MSP}}$ & AUROC$_{\mathrm{MI}}$ & AURC & Acc@80\% & AUROC$_{\mathrm{MSP}}$ & AUROC$_{\mathrm{MI}}$ & AURC & Acc@80\%\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# Table: per-class SDSS recall vs DECaLS recall (main models)
MAIN = [m for m in ["resnet50", "vit_small", "effnet_b0_imnet_full", "zoobot_full", "zoobot"] if m in ORDER]
q = pc.dropna(subset=["sdss_recall"])
cls_order = ["Round Smooth", "In-between Smooth", "Cigar Smooth", "Unbarred Tight Spiral", "Unbarred Loose Spiral",
             "Edge-on without Bulge", "Edge-on with Bulge"]
rows = []
for c in cls_order:
    x = q[q.cls == c]; ns = int(x.n_sdss.iloc[0]); nd = int(x.n_decals.iloc[0])
    cells = [f"{x[x.model==m].decals_recall.iloc[0]:.1f} / {x[x.model==m].sdss_recall.iloc[0]:.1f}" for m in MAIN]
    rows.append(f"{c} & {nd} / {ns:,} & " + " & ".join(cells) + " \\\\")
hdr = " & ".join({"resnet50": "ResNet-50", "vit_small": "ViT-S/16", "effnet_b0_imnet_full": "EffNet-B0 IN (A)",
                  "zoobot_full": "Zoobot (A)", "zoobot": "Zoobot (B)"}[m] for m in MAIN)
write("tab_perclass.tex", r"""\begin{table*}[t]
\caption{Per-class recall (\%) on the DECaLS test images / SDSS images of the seven shared classes (restricted protocol). The second column gives the number of images in each set}
\label{tab:perclass}
\centering
\begin{tabular}{@{}ll""" + "c" * len(MAIN) + r"""@{}}
\toprule
Class & $n$ (DECaLS / SDSS) & """ + hdr + r"""\\
\midrule
""" + "\n".join(rows).replace(",", "{,}") + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# Table: per-class F1 on DECaLS (10 classes)
q = pc.dropna(subset=["f1_10"])
cls10 = ["Disturbed", "Merging", "Round Smooth", "In-between Smooth", "Cigar Smooth", "Barred Spiral",
         "Unbarred Tight Spiral", "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge"]
rows = []
for c in cls10:
    v = [q[(q.cls == c) & (q.model == m)].f1_10.iloc[0] for m in ORDER]; kb = int(np.argmax(v))
    rows.append(c + " & " + " & ".join((f"\\textbf{{{x:.1f}}}" if i == kb else f"{x:.1f}") for i, x in enumerate(v)) + " \\\\")
hdr = " & ".join({"resnet50": "ResNet-50", "vit_small": "ViT-S/16", "effnet_b0_imnet_full": "EffNet-B0 IN (A)",
                  "zoobot_full": "Zoobot (A)", "effnet_b0_imnet": "EffNet-B0 IN (B)", "zoobot": "Zoobot (B)"}[m] for m in ORDER)
write("tab_f1.tex", r"""\begin{table*}[t]
\caption{Per-class F1 score (\%) on the DECaLS test split. The best value in each row is in bold}
\label{tab:f1}
\centering
\begin{tabular}{@{}l""" + "c" * len(ORDER) + r"""@{}}
\toprule
Class & """ + hdr + r"""\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# paired tests
rows = []
for _, x in t.iterrows():
    p = "$<10^{-4}$" if x.p < 1e-4 else f"{x.p:.4f}"
    if x.a == "effnet_b0_imnet" and x.b == "resnet50": continue
    n01, n10 = f"{int(x.n01):,}".replace(",", "{,}"), f"{int(x.n10):,}".replace(",", "{,}")
    rows.append(f"{NM[x.a]} vs {NM[x.b]} & {x['diff']:+.2f} & [{x.lo:+.2f}, {x.hi:+.2f}] & {n01} & {n10} & {p} \\\\")
write("tab_tests.tex", r"""\begin{table*}[t]
\caption{Paired comparisons on the 17{,}401 SDSS images (restricted protocol). $n_{01}$: images that only the first model classifies correctly; $n_{10}$: images that only the second model classifies correctly. The interval is a paired bootstrap 95\,\% interval of the accuracy difference and $p$ is from McNemar's test}
\label{tab:tests}
\centering
\begin{tabular}{@{}lccccc@{}}
\toprule
Comparison & $\Delta$Acc & 95\,\% CI & $n_{01}$ & $n_{10}$ & $p$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# Grad-CAM mass-in-galaxy
mg = pd.read_csv("results/mig.csv")
rows = []
for m in ORDER:
    x = mg[mg.model == m]
    if x.empty: continue
    d, s = x[x.set == "decals"], x[x.set == "sdss"]
    rows.append(f"{NM[m]} & {d.mig.mean():.3f} ({d.mig.std():.3f}) & {d.area.mean():.3f} & {s.mig.mean():.3f} ({s.mig.std():.3f}) & {s.area.mean():.3f} \\\\")
write("tab_mig.tex", r"""\begin{table*}[t]
\caption{Share of Grad-CAM mass inside the galaxy mask (MiG, Eq.~(\ref{eq:mig}); mean and standard deviation) and the mean fraction of the image covered by the mask, which is the MiG of a map that ignores the galaxy. DECaLS: 300 test images, 30 per class; SDSS: 300 randomly drawn images}
\label{tab:mig}
\centering
\begin{tabular}{@{}lcccc@{}}
\toprule
 & \multicolumn{2}{c}{DECaLS} & \multicolumn{2}{c}{SDSS}\\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Model & MiG & Mask area & MiG & Mask area\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
print("tables written", ORDER)
