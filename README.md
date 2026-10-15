# Galaxy morphology classification under cross-survey shift (DECaLS to SDSS)

Code for a study of how well galaxy morphology classifiers hold up when they are
trained on one survey and applied to another. Models are trained on Galaxy10
DECaLS only and then evaluated, with no adaptation, on (i) the held-out DECaLS
test split, (ii) the same test images degraded to SDSS resolution, and (iii)
Galaxy10 SDSS. Besides accuracy we look at calibration (ECE, NLL, Brier,
temperature scaling), MC dropout uncertainty, selective prediction and Grad-CAM
maps.

## What the code does

- `src/train_eval.py` trains the models on DECaLS (70/15/15 stratified split,
  random_state 42), keeps the best-validation checkpoint, then writes logits and
  30-pass MC dropout outputs for the DECaLS validation and test sets, the
  low-resolution DECaLS test set and the full SDSS set. It also measures
  single-image latency and, for seed 42, Grad-CAM maps and weights.
- `analysis/` works only on the saved predictions: metrics, calibration,
  ensembles, paired tests, figures, and the mass-in-galaxy score of the Grad-CAM maps.

Model kinds:

| kind | encoder | recipe |
| --- | --- | --- |
| `resnet50` | ResNet-50, ImageNet | 30 epochs full fine-tuning, lr 1e-4 |
| `vit_small` | ViT-S/16, ImageNet | 30 epochs full fine-tuning, lr 1e-4 |
| `effnet_b0_imnet` | EfficientNet-B0, ImageNet | 10 epochs head only (lr 1e-3), then 20 epochs full (lr 1e-5) |
| `zoobot` | Zoobot EfficientNet-B0 | same two-stage recipe |
| `effnet_b0_imnet_full` | EfficientNet-B0, ImageNet | 30 epochs full fine-tuning, lr 1e-4 |
| `zoobot_full` | Zoobot EfficientNet-B0 | 30 epochs full fine-tuning, lr 1e-4 |

The optimiser is AdamW (weight decay 1e-4) with a cosine schedule, label
smoothing 0.1, batch size 32, mixed precision, and GPU rotation / flip /
brightness / contrast augmentation.

### Class mapping

Only seven classes exist in both surveys. SDSS class index to DECaLS class index:

    {SDSS 1 -> DECaLS 2, 2 -> 3, 3 -> 4, 7 -> 6, 9 -> 7, 6 -> 8, 4 -> 9}

For the shared-class results the 10-way softmax is restricted to these seven
DECaLS classes and renormalised.

## Steps

All commands are run from the repository root.

1. Install (a CUDA GPU is needed for training; the analysis runs on CPU):

        pip install -r requirements.txt

2. Download the data (several GB, goes to `data/`):

        bash scripts/download_data.sh

3. Train and predict. This takes about 1.4 h for four models (resnet50,
   vit_small, effnet_b0_imnet, zoobot) on one Tesla T4. The two `_full`
   variants are extra runs.

        python src/train_eval.py --models resnet50,vit_small,effnet_b0_imnet,zoobot --seed 42 --out runs/seed42
        python src/train_eval.py --models effnet_b0_imnet_full,zoobot_full --seed 42 --out runs/seed42

   Run `python src/train_eval.py --help` for all options. The data split always
   uses random_state 42; `--seed` only changes training. Grad-CAM maps and weights
   are written when the seed is 42 (or when `--save-cams` is given). Other seeds
   can go into their own folder, for example `--seed 1 --out runs/seed1`.

4. Analyse (tables go to `results/`):

        python analysis/analyse.py runs/seed42

5. Figures (go to `figs/`; needs the csv/npy files from step 4 and the data files):

        python analysis/figures.py
        python analysis/diagrams.py

6. Mass-in-galaxy score of the Grad-CAM maps (needs both h5 files in `data/`):

        python analysis/mig.py runs/seed42

The run folder can also be given with the `RUN` environment variable. The
variables `DATA`, `RESULTS` and `FIGS` change the data, table and figure folders.
Models whose prediction files are missing are skipped.

## Running on Kaggle

Attach the datasets `ijlalahmed10/galaxy10-decals` and `eliasillescas/galaxy10`,
enable a GPU and internet (internet is needed to fetch the pretrained weights),
and run

    python kaggle/run_kaggle.py --models resnet50,zoobot --seed 42

`run_kaggle.py` finds the h5 files under `/kaggle/input` and writes to
`/kaggle/working`. To push a kernel with the Kaggle CLI, copy
`kaggle/kernel-metadata.example.json` to `kernel-metadata.json` and set your own
`id` (your Kaggle username and a kernel slug) and `title`. The `id` is left as a
placeholder on purpose because it has to be your own account.

## Folder layout

    src/train_eval.py          training and inference
    analysis/analyse.py        metrics, calibration, ensembles, tests
    analysis/analyse_lib.py    shared helpers for the figure script
    analysis/figures.py        result figures
    analysis/diagrams.py       pipeline and model block diagrams
    analysis/mig.py            mass-in-galaxy score for Grad-CAM
    scripts/download_data.sh   data download
    kaggle/                    Kaggle wrapper and metadata example
    data/ runs/ results/ figs/ created when you run things (not tracked)

## Hardware and reproducibility

The results in the paper were produced on one Tesla T4 with timm 1.0.29, seed
42 and mixed precision (fp16 autocast). GPU kernels and cuDNN benchmark mode are
not fully deterministic, so a rerun can differ slightly even with the same seed.
Pretrained weights are downloaded at run time.

## Citation

The accompanying paper is under review. A citation will be added here once it
is published.

## License

MIT, see `LICENSE`.

## Results used in the paper

`results_paper/` holds the result tables produced by `analysis/analyse.py` and `analysis/mig.py`
for the seed-42 runs reported in the paper, together with `split.npz` (the exact train, validation
and test indices and the labels of both datasets). They let the numbers in the paper be checked
without a GPU. `analysis/tables.py` turns these CSV files into the LaTeX tables of the paper.

| File | Content |
|---|---|
| `per_seed.csv` | accuracy, macro-F1, balanced accuracy, ECE, NLL, Brier score, AUROC, AURC and temperatures for every model and test condition |
| `bootstrap.csv` | 95% bootstrap intervals of accuracy and ECE |
| `paired_tests.csv` | paired bootstrap intervals and McNemar tests on SDSS |
| `per_class.csv` | per-class recall and F1 |
| `mig.csv` | Grad-CAM mass-in-galaxy score per image |

`results_paper/predictions/` holds, for each of the six trained models (seed 42), the logits, the
Monte Carlo dropout mean probabilities and the per-image uncertainty scores (predictive entropy,
mutual information and summed variance) on the DECaLS validation and test splits, the degraded
DECaLS test split and the full Galaxy10 SDSS set, with the training log of each run. To recompute
all tables without training, run from the repository root

    RUN=results_paper/predictions python analysis/analyse.py results_paper/predictions

(the folder already contains a copy of `split.npz`). The Grad-CAM score
(`mig.py`) needs the full prediction files, which are produced by `src/train_eval.py`.
