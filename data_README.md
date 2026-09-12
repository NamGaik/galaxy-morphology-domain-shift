# Data

This project uses two publicly available astronomical imaging datasets. Neither
dataset is stored in this repository directly (see rationale below); both are
accessed programmatically via the `astroNN` Python package.

## Galaxy10 DECaLS (training/source survey)

- 17,736 colour images, 256x256x3, 10 morphology classes
- Images from DESI Legacy Imaging Surveys (DECaLS); labels from Galaxy Zoo
- Permanently archived on Zenodo: https://zenodo.org/records/10845026
- Documentation: https://astronn.readthedocs.io/en/latest/galaxy10.html
- Loaded via:
  ```python
  from astroNN.datasets import load_galaxy10
  images, labels = load_galaxy10()
  ```

## Galaxy10 SDSS (target/cross-survey evaluation)

- 21,785 colour images, 69x69x3, 10 morphology classes
- Images from Sloan Digital Sky Survey (SDSS); labels from Galaxy Zoo
- Documentation: https://astronn.readthedocs.io/en/latest/galaxy10sdss.html
- Loaded via:
  ```python
  from astroNN.datasets import load_galaxy10sdss
  images, labels = load_galaxy10sdss()
  ```

## Augmented training set (frozen snapshot)

A frozen, one-time snapshot of the augmented Galaxy10 DECaLS training set (seed=42,
matching the exact split and augmentation pipeline described in the paper's
Methodology) is archived separately on Zenodo due to file size:

**[INSERT ZENODO LINK HERE]**

This file was generated using `src/save_augmented_dataset.py` in this repository.

## Why the raw datasets aren't stored in this repository

- The original DECaLS and SDSS datasets are already permanently archived by their
  maintainers at the links above; duplicating them here would be redundant.
- GitHub blocks any single file over 100MB, and both source datasets exceed this
  well before compression.
- Using the canonical sources ensures anyone reproducing this work pulls from the
  same authoritative, version-stable data referenced in the paper.

## Reproducing the exact data split

All splits use `random_state=42` / `SEED=42`, applied via a stratified 70/15/15
split (train/val/test) on Galaxy10 DECaLS, and a 20/80 calibration/held-out split
on the filtered Galaxy10 SDSS set. See `src/train.py` and the main notebook for
the exact split code.
