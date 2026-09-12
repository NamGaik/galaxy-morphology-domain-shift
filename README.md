# Deep Learning for Galaxy Morphology Classification Under Cross-Survey Domain Shift

Code and reproducibility artifacts for the paper evaluating ResNet-50, ViT-Small, and Zoobot
on galaxy morphology classification, with a focus on predictive uncertainty, calibration,
and robustness under zero-shot cross-survey transfer from Galaxy10 DECaLS to Galaxy10 SDSS.

## Repository structure

```
├── notebooks/
│   └── main_experiment.ipynb      Full experimental pipeline (Colab)
├── src/
│   ├── train.py                   Training loop, evaluation, Zoobot two-stage training
│   ├── evaluate.py                Baseline metrics, confusion matrices
│   ├── mc_dropout.py              Monte Carlo Dropout uncertainty estimation
│   ├── calibration.py             ECE, reliability diagrams, temperature scaling
│   ├── selective_prediction.py    Coverage-accuracy curves, label efficiency
│   ├── gradcam.py                 Grad-CAM / Grad-CAM++ interpretability
│   └── save_augmented_dataset.py  Frozen snapshot of the augmented training set
├── figures/                       Fig1.png ... Fig17.png as used in the paper
├── results/                       Tables 1-9 as exported CSV files
└── data/
    └── README.md                  Dataset access and provenance
```

## Reproducing the results

1. Install dependencies: `pip install -r requirements.txt`
2. Run `notebooks/main_experiment.ipynb`, or the individual scripts in `src/` in sequence
3. A fixed random seed (`SEED=42`) is used throughout for the data split, training, and
   augmentation, so results are deterministic given the same environment

## Datasets

- **Galaxy10 DECaLS** -- permanently archived at https://zenodo.org/records/10845026
- **Galaxy10 SDSS** -- accessed via `astroNN` (https://astronn.readthedocs.io/en/latest/galaxy10sdss.html)

Both are loaded programmatically via the `astroNN` package; see `data/README.md` for details.

## Citation

If you use this code, please cite the associated paper (details to be added upon publication).

## License

MIT License -- see `LICENSE` for details.
