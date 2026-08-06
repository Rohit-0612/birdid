#!/usr/bin/env python3
"""
Train a CUB-200-2011 classifier from the committed official split.

    python3 train.py --epochs 35                       # Colab T4/L4
    python3 train.py --epochs 3 --limit-train 500      # local smoke test
    python3 train.py --eval-only --resume best.pth     # measure a checkpoint

Why this exists
---------------
The checkpoint previously shipped in this repo cannot be honestly evaluated.
Measured with it: official CUB train 94.6%, official test 94.2%, and official
test images never present in birds_split/train 95.2% -- images it supposedly
never saw scoring HIGHER than ones it did, against a realistic ~86-88% ceiling
for this architecture on CUB. The split it was trained on was never recorded.

This script only ever reads data/splits/official_{train,val,test}.txt, records
their sha256 in the checkpoint, and never touches the test manifest except in
a final explicit evaluation. That makes every number it produces reproducible
and attributable.

Recipe: RandAugment + RandomResizedCrop + hflip, label smoothing 0.1,
mixup/cutmix, AdamW with discriminative learning rates, cosine schedule with
warmup, EMA weights, AMP. Trains at 300px and evaluates at 380px (FixRes,
Touvron et al. 2019) -- faster per epoch and usually slightly more accurate.

Expected: 84-88% top-1 on official CUB test. Treat >90% from this recipe as a
leak to investigate, not a win.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CUB_IMAGES = os.path.join(PROJECT_DIR, "CUB_200_2011", "images")
SPLIT_DIR = os.path.join(PROJECT_DIR, "data", "splits")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Data ─────────────────────────────────────────────────────────────────

class CUBManifest(Dataset):
    """Reads 'relative/path.jpg<TAB>class_index' lines; images stay in place."""

    def __init__(self, manifest_path, transform, root=CUB_IMAGES, limit=None):
        self.samples = []
        with open(manifest_path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                rel, cls = line.split("\t")
                self.samples.append((os.path.join(root, rel), int(cls)))
        if limit:
            self.samples = self.samples[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, cls = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), cls


def build_transforms(train_size, eval_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(train_size, scale=(0.5, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, value="random"),
    ])
    # Evaluate at higher resolution than training (FixRes). Aspect-preserving
    # resize then centre crop -- the standard eval pipeline.
    eval_tf = transforms.Compose([
        transforms.Resize(int(eval_size * 1.15),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(eval_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


# ── Model ────────────────────────────────────────────────────────────────

def build_model(num_classes, device):
    m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, num_classes)
    return m.to(device)


def param_groups(model, backbone_lr, head_lr, weight_decay):
    """Discriminative LRs: the pretrained trunk moves slower than the new head.

    Norm layers and biases are excluded from weight decay -- decaying them is
    a small but consistent loss (Bag of Tricks, He et al. 2019).
    """
    head_ids = {id(p) for p in model.classifier.parameters()}
    groups = {"bb_decay": [], "bb_nodecay": [], "hd_decay": [], "hd_nodecay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or name.endswith(".bias")
        key = ("hd_" if id(p) in head_ids else "bb_") + ("nodecay" if no_decay else "decay")
        groups[key].append(p)
    return [
        {"params": groups["bb_decay"], "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": groups["bb_nodecay"], "lr": backbone_lr, "weight_decay": 0.0},
        {"params": groups["hd_decay"], "lr": head_lr, "weight_decay": weight_decay},
        {"params": groups["hd_nodecay"], "lr": head_lr, "weight_decay": 0.0},
    ]


def cosine_with_warmup(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ── Eval ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, tta=False):
    model.eval()
    correct = top5 = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        if tta:
            # Average logits, not probabilities: softmax is not linear, so
            # averaging probabilities dilutes a confident view with a vague one.
            logits = (logits + model(torch.flip(x, dims=[3]))) / 2
        _, pred5 = logits.topk(5, dim=1)
        correct += (pred5[:, 0] == y).sum().item()
        top5 += (pred5 == y[:, None]).any(dim=1).sum().item()
        total += y.numel()
    return 100 * correct / total, 100 * top5 / total


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-size", type=int, default=300)
    ap.add_argument("--eval-size", type=int, default=380)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--ema-decay", type=float, default=0.9998)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-mix", action="store_true", help="disable mixup/cutmix")
    ap.add_argument("--limit-train", type=int, help="smoke-test on N training images")
    ap.add_argument("--limit-eval", type=int,
                    help="smoke-test: cap val/test images. Never use for a real run — "
                         "a partial test set is not the benchmark.")
    ap.add_argument("--out", default=os.path.join(PROJECT_DIR, "bird_model_official.pth"))
    ap.add_argument("--resume")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    classes = [l.split(" ", 1)[1].strip()
               for l in open(os.path.join(PROJECT_DIR, "CUB_200_2011", "classes.txt"))]
    manifests = {s: os.path.join(SPLIT_DIR, f"official_{s}.txt")
                 for s in ("train", "val", "test")}
    for s, p in manifests.items():
        if not os.path.exists(p):
            raise SystemExit(f"Missing {p}. Run: python3 scripts/make_split.py")

    train_tf, eval_tf = build_transforms(args.train_size, args.eval_size)
    ds = {
        "train": CUBManifest(manifests["train"], train_tf, limit=args.limit_train),
        "val": CUBManifest(manifests["val"], eval_tf, limit=args.limit_eval),
        "test": CUBManifest(manifests["test"], eval_tf, limit=args.limit_eval),
    }
    if args.limit_eval or args.limit_train:
        print("SMOKE TEST: data is truncated, resulting numbers are meaningless")
    pin = device.type == "cuda"
    loaders = {
        k: DataLoader(v, batch_size=args.batch_size, shuffle=(k == "train"),
                      num_workers=args.workers, pin_memory=pin,
                      persistent_workers=args.workers > 0, drop_last=(k == "train"))
        for k, v in ds.items()
    }
    print(f"device={device}  train={len(ds['train'])}  val={len(ds['val'])}  "
          f"test={len(ds['test'])}  classes={len(classes)}")
    print(f"train@{args.train_size}px  eval@{args.eval_size}px (FixRes)")

    model = build_model(len(classes), device)
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        print(f"resumed from {args.resume}")

    if args.eval_only:
        for split in ("val", "test"):
            a1, a5 = evaluate(model, loaders[split], device)
            t1, t5 = evaluate(model, loaders[split], device, tta=True)
            print(f"{split:<5} top-1 {a1:.2f}%  top-5 {a5:.2f}%   "
                  f"| +hflip-TTA top-1 {t1:.2f}%  top-5 {t5:.2f}%")
        return 0

    optimizer = torch.optim.AdamW(
        param_groups(model, args.backbone_lr, args.head_lr, args.weight_decay))
    steps = max(1, len(loaders["train"]))
    scheduler = cosine_with_warmup(optimizer, args.warmup_epochs, args.epochs, steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    mix = None
    if not args.no_mix:
        try:
            from torchvision.transforms import v2
            mix = v2.RandomChoice([v2.MixUp(alpha=0.2, num_classes=len(classes)),
                                   v2.CutMix(alpha=1.0, num_classes=len(classes))])
        except Exception as e:                                   # noqa: BLE001
            print(f"mixup/cutmix unavailable ({e}); continuing without")

    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -1 rather than 0: the first epoch must always save, otherwise a run whose
    # val accuracy never exceeds 0% leaves no checkpoint and the final reload
    # fails on a missing file.
    best_val, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for i, (x, y) in enumerate(loaders["train"]):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if mix is not None:
                x, y = mix(x, y)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update_parameters(model)
            running += loss.item() * x.size(0)
            seen += x.size(0)
            if (i + 1) % 40 == 0:
                print(f"  epoch {epoch} step {i+1}/{steps} loss {running/seen:.3f}")

        raw1, _ = evaluate(model, loaders["val"], device)
        ema1, _ = evaluate(ema.module, loaders["val"], device)
        use_ema = ema1 >= raw1
        val1 = max(raw1, ema1)
        history.append({"epoch": epoch, "train_loss": running / max(1, seen),
                        "val_top1_raw": raw1, "val_top1_ema": ema1})
        print(f"epoch {epoch:>2}/{args.epochs}  loss {running/max(1,seen):.3f}  "
              f"val {raw1:.2f}% (ema {ema1:.2f}%)  "
              f"lr {optimizer.param_groups[0]['lr']:.2e}  {time.time()-t0:.0f}s")

        if val1 > best_val:
            best_val = val1
            src = ema.module if use_ema else model
            best_state = copy.deepcopy(src.state_dict())
            torch.save({
                "model_state_dict": best_state,
                "class_names": classes,
                "epoch": epoch,
                "val_acc": best_val,
                "used_ema": use_ema,
                "arch": "efficientnet_v2_s",
                "train_size": args.train_size,
                "eval_size": args.eval_size,
                "split_manifest_sha256": {k: sha256(v) for k, v in manifests.items()},
                "split_source": "data/splits/official_*.txt (official CUB split)",
                "recipe": vars(args),
                "history": history,
            }, args.out)
            print(f"    saved (best val {best_val:.2f}%) -> {args.out}")

    # Test is touched exactly once, after all model selection is finished.
    if best_state is not None:
        model.load_state_dict(best_state)
    t1, t5 = evaluate(model, loaders["test"], device)
    tta1, tta5 = evaluate(model, loaders["test"], device, tta=True)
    print(f"\nOFFICIAL CUB TEST  top-1 {t1:.2f}%  top-5 {t5:.2f}%")
    print(f"  + hflip TTA      top-1 {tta1:.2f}%  top-5 {tta5:.2f}%")
    if t1 > 90:
        print("  WARNING: >90% from this recipe suggests a leak. Investigate "
              "before reporting it.")

    ck = torch.load(args.out, map_location="cpu")
    ck.update(test_top1=t1, test_top5=t5, test_top1_tta=tta1, test_top5_tta=tta5)
    torch.save(ck, args.out)

    # results.json must only ever contain numbers from a complete run on the
    # full manifests. A truncated smoke test writing into it is precisely the
    # kind of unearned figure this project already had to remove once.
    if args.limit_train or args.limit_eval:
        print("smoke test — results.json not written")
        return 0

    results_path = os.path.join(PROJECT_DIR, "evaluation", "results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    results["EfficientNetV2-S (official split)"] = {
        "top1": round(t1, 2), "top5": round(t5, 2),
        "top1_tta": round(tta1, 2), "val_top1": round(best_val, 2),
        "split": "official CUB-200-2011 (data/splits/official_*.txt)",
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "epochs": args.epochs,
    }
    json.dump(results, open(results_path, "w"), indent=2)
    print(f"results -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
