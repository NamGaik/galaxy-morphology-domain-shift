"""Train and evaluate galaxy classifiers on Galaxy10 DECaLS and run inference on
the DECaLS test split, a low-resolution copy of it, and Galaxy10 SDSS.

Model kinds: resnet50, vit_small, effnet_b0_imnet, zoobot (two-stage recipe) and
effnet_b0_imnet_full, zoobot_full (same 30-epoch full fine-tuning as resnet50).
"""
import argparse, os, time, json, math, tempfile
import numpy as np, h5py, torch, torch.nn as nn, torch.nn.functional as F
import timm
from sklearn.model_selection import train_test_split

ALL_KINDS = "resnet50,vit_small,effnet_b0_imnet,zoobot,effnet_b0_imnet_full,zoobot_full"
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--models", default=ALL_KINDS, help="comma separated list of model kinds")
ap.add_argument("--seed", type=int, default=42, help="training seed (the data split always uses 42)")
ap.add_argument("--decals", default="data/Galaxy10_DECals.h5", help="path to Galaxy10_DECals.h5")
ap.add_argument("--sdss", default="data/Galaxy10.h5", help="path to Galaxy10.h5 (SDSS)")
ap.add_argument("--out", default="runs/seed42", help="output directory")
ap.add_argument("--save-cams", action="store_true",
                help="save Grad-CAM maps and weights even if seed != 42 (they are always saved for seed 42)")
args = ap.parse_args()

SEEDS = [args.seed]
MODELS = [m.strip() for m in args.models.split(",") if m.strip()]
OUT = args.out; os.makedirs(OUT, exist_ok=True)
BEST_PATH = os.path.join(tempfile.gettempdir(), "best.pth")  # best-validation checkpoint
dev = "cuda"
torch.backends.cudnn.benchmark = True
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
print("GPU:", torch.cuda.get_device_name(0), "timm", timm.__version__, flush=True)


def load_h5(path):
    with h5py.File(path, "r") as f:
        return f["images"][()], f["ans"][()].astype(np.int64)


def resize_set(arr, low=False, bs=512):
    out = torch.empty((len(arr), 3, 224, 224), dtype=torch.uint8)
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(arr[i:i + bs]).to(dev).permute(0, 3, 1, 2).float()
        if low:  # DECaLS 256 px -> 69 px (SDSS cut-out size) -> 224 px
            x = F.interpolate(x, size=(69, 69), mode="bilinear", align_corners=False, antialias=True)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        out[i:i + bs] = x.round().clamp(0, 255).to(torch.uint8).cpu()
    return out


t0 = time.time()
dimg, dlab = load_h5(args.decals)
simg, slab = load_h5(args.sdss)
print("DECaLS", dimg.shape, np.bincount(dlab), "SDSS", simg.shape, np.bincount(slab), flush=True)
idx = np.arange(len(dlab))
idx_temp, idx_test = train_test_split(idx, test_size=0.15, random_state=42, stratify=dlab)
idx_train, idx_val = train_test_split(idx_temp, test_size=0.1765, random_state=42, stratify=dlab[idx_temp])
np.savez(f"{OUT}/split.npz", train=idx_train, val=idx_val, test=idx_test, dlab=dlab, slab=slab)
D = resize_set(dimg); Dlow = resize_set(dimg[idx_test], low=True); S = resize_set(simg)
del dimg, simg
print("preprocessed", round(time.time() - t0), "s", flush=True)


def gpu_aug(x):
    B = x.shape[0]
    th = torch.rand(B, device=dev) * 2 * math.pi
    fx = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    fy = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    A = torch.zeros(B, 2, 3, device=dev)
    A[:, 0, 0] = torch.cos(th) * fx; A[:, 0, 1] = -torch.sin(th) * fy
    A[:, 1, 0] = torch.sin(th) * fx; A[:, 1, 1] = torch.cos(th) * fy
    g = F.affine_grid(A, list(x.shape), align_corners=False)
    x = F.grid_sample(x, g, mode="bilinear", padding_mode="zeros", align_corners=False)
    b = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 0.4 - 0.2)
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 0.4 - 0.2)
    x = (x * b).clamp(0, 1)
    m = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).mean((1, 2)).view(B, 1, 1, 1)
    return ((x - m) * c + m).clamp(0, 1)


def prep(xb, aug=False):
    x = xb.to(dev, non_blocking=True).float() / 255.0
    if aug:
        x = gpu_aug(x)
    return (x - MEAN) / STD


class Net(nn.Module):
    def __init__(self, kind):
        super().__init__()
        name = {"resnet50": "resnet50", "vit_small": "vit_small_patch16_224",
                "effnet_b0_imnet": "efficientnet_b0",
                "zoobot": "hf_hub:mwalmsley/zoobot-encoder-efficientnet_b0"}[kind.replace("_full", "")]
        self.kind = kind
        self.encoder = timm.create_model(name, pretrained=True, num_classes=0)
        d = self.encoder.num_features
        if kind.replace("_full", "") in ("zoobot", "effnet_b0_imnet"):
            self.head = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 10))
        else:
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 10))

    def forward(self, x):
        return self.head(self.encoder(x))


@torch.no_grad()
def feats(model, X, ids=None, bs=256):
    model.eval(); out = []
    n = len(X) if ids is None else len(ids)
    for i in range(0, n, bs):
        xb = X[i:i + bs] if ids is None else X[ids[i:i + bs]]
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model.encoder(prep(xb)).float())
    return torch.cat(out)


@torch.no_grad()
def logits(model, X, ids=None):
    f = feats(model, X, ids); model.eval()
    return model.head(f).cpu()


def run_epochs(model, params, lr, epochs, tag, seed):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler()
    g = torch.Generator().manual_seed(seed)
    best, hist = -1, []
    ytr = torch.from_numpy(dlab[idx_train]); yv = torch.from_numpy(dlab[idx_val])
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(idx_train), generator=g); tl = 0; n = 0
        for i in range(0, len(perm) - 31, 32):
            b = perm[i:i + 32]
            x = prep(D[idx_train[b.numpy()]], aug=True); y = ytr[b].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(b); n += len(b)
        sch.step()
        lv = logits(model, D, idx_val)
        va = (lv.argmax(1) == yv).float().mean().item(); vl = F.cross_entropy(lv, yv).item()
        hist.append(dict(stage=tag, epoch=ep + 1, train_loss=tl / n, val_loss=vl, val_acc=va))
        if va > best:
            best = va; torch.save(model.state_dict(), BEST_PATH)
        print(tag, ep + 1, round(tl / n, 4), round(vl, 4), round(va, 4), flush=True)
    model.load_state_dict(torch.load(BEST_PATH))
    return hist


@torch.no_grad()
def mc(model, f, T=30):
    model.head.train(); ps = []
    for _ in range(T):
        ps.append(F.softmax(model.head(f), 1))
    model.eval(); P = torch.stack(ps)
    pm = P.mean(0)
    H = -(pm * pm.clamp_min(1e-12).log()).sum(1)
    EH = -(P * P.clamp_min(1e-12).log()).sum(2).mean(0)
    var = P.var(0).sum(1)
    return pm.cpu().numpy().astype(np.float32), torch.stack([H, H - EH, var], 1).cpu().numpy().astype(np.float32)


def latency(model):
    model.eval(); x = torch.randn(1, 3, 224, 224, device=dev)
    with torch.no_grad():
        for _ in range(30):
            model(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(200):
            model(x)
        torch.cuda.synchronize()
    return (time.time() - t) / 200 * 1000


def gradcam(model, X, ids):
    enc = model.encoder
    layer = {"resnet50": lambda: enc.layer4, "vit_small": lambda: enc.blocks[-1].norm1}.get(
        model.kind, lambda: enc.conv_head)()
    store = {}
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    model.eval(); cams = []
    for i in ids:
        x = prep(X[int(i):int(i) + 1])
        out = model(x); a = store["a"]
        c = out.argmax(1).item()
        g, = torch.autograd.grad(out[0, c], a)
        if a.dim() == 3:  # ViT tokens -> 14 x 14 grid
            a = a[:, enc.num_prefix_tokens:, :].transpose(1, 2).reshape(1, -1, 14, 14)
            g = g[:, enc.num_prefix_tokens:, :].transpose(1, 2).reshape(1, -1, 14, 14)
        cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        cams.append((cam / (cam.max() + 1e-8)).detach().cpu().numpy().astype(np.float16))
    h.remove()
    return np.stack(cams)


for seed in SEEDS:
    for kind in MODELS:
        torch.manual_seed(seed); np.random.seed(seed)
        t1 = time.time(); model = Net(kind).to(dev)
        nparam = sum(p.numel() for p in model.parameters()) / 1e6
        if kind in ("zoobot", "effnet_b0_imnet"):
            for p in model.encoder.parameters():
                p.requires_grad = False
            h = run_epochs(model, model.head.parameters(), 1e-3, 10, "stage1", seed)
            for p in model.encoder.parameters():
                p.requires_grad = True
            h += run_epochs(model, model.parameters(), 1e-5, 20, "stage2", seed)
        else:
            h = run_epochs(model, model.parameters(), 1e-4, 30, "full", seed)
        res = {"kind": kind, "seed": seed, "params_M": nparam, "train_min": (time.time() - t1) / 60,
               "gpu": torch.cuda.get_device_name(0), "hist": h, "latency_ms": latency(model)}
        arrs = {}
        for nm, X, ids in [("val", D, idx_val), ("test", D, idx_test), ("lowres", Dlow, None), ("sdss", S, None)]:
            f = feats(model, X, ids); model.eval()
            with torch.no_grad():
                arrs[f"logit_{nm}"] = model.head(f).cpu().numpy().astype(np.float32)
            pm, u = mc(model, f); arrs[f"mc_{nm}"] = pm; arrs[f"unc_{nm}"] = u
        if seed == 42 or args.save_cams:
            try:
                rng = np.random.default_rng(0)
                tid = np.concatenate([rng.choice(np.where(dlab[idx_test] == c)[0], 30, replace=False) for c in range(10)])
                sid = rng.choice(len(S), 300, replace=False)
                arrs["cam_test_pos"] = tid; arrs["cam_test"] = gradcam(model, D, idx_test[tid])
                arrs["cam_sdss_pos"] = sid; arrs["cam_sdss"] = gradcam(model, S, sid)
            except Exception as e:
                print("gradcam failed", kind, repr(e), flush=True)
            torch.save(model.state_dict(), f"{OUT}/weights_{kind}_s{seed}.pth")
        np.savez_compressed(f"{OUT}/pred_{kind}_s{seed}.npz", **arrs)
        json.dump(res, open(f"{OUT}/res_{kind}_s{seed}.json", "w"))
        print("DONE", kind, seed, round(res["train_min"], 1), "min", "acc",
              (arrs["logit_test"].argmax(1) == dlab[idx_test]).mean(), flush=True)
        del model; torch.cuda.empty_cache()
print("ALL DONE", round((time.time() - t0) / 3600, 2), "h")
