"""Kaggle wrapper: find the two .h5 files under /kaggle/input and run src/train_eval.py.

Usage inside a Kaggle script kernel (with this repo uploaded next to this file):
    python run_kaggle.py --models resnet50,zoobot --seed 42
Extra arguments are passed on to train_eval.py unchanged.
"""
import glob, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "..", "src", "train_eval.py")
if not os.path.exists(TRAIN):  # kernel with all files in one folder
    TRAIN = os.path.join(HERE, "src", "train_eval.py")


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        sys.exit(f"could not find {name} under /kaggle/input")
    return hits[0]


# timm is normally preinstalled on Kaggle; the paper used 1.0.29
try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm==1.0.29"], check=True)

cmd = [sys.executable, TRAIN, "--decals", find("Galaxy10_DECals.h5"), "--sdss", find("Galaxy10.h5"),
       "--out", "/kaggle/working"] + sys.argv[1:]
print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
