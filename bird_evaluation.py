# ============================================================
#  BIRD PROJECT — Research Evaluation Suite
#  Includes:
#    1. Full Evaluation Metrics (Precision, Recall, F1, Confusion Matrix)
#    2. Model Comparison Table
#    3. Grad-CAM Visualisation
#    4. Confidence Calibration
#
#  HOW TO RUN:
#    Step 1: Make sure bird_identifier_local.py is in the same folder
#    Step 2: Install extra libraries:
#            pip3 install grad-cam scikit-learn matplotlib seaborn
#    Step 3: Run: python3 bird_evaluation.py
#    Step 4: All results saved to BirdProject/evaluation/ folder
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

# ============================================================
# ✅ SETTINGS — update these paths to match your setup
# ============================================================
MODEL_PATH   = "/Users/swayam/Desktop/birdsproject/best_bird_model.pth"
TEST_DIR     = "/Users/swayam/Desktop/birdsproject/birds_split/test"
SAVE_DIR     = "/Users/swayam/Desktop/birdsproject/evaluation"
BATCH_SIZE   = 32
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nRunning on: {device}")

# ── Load model ───────────────────────────────────────────────
print("Loading model...")
checkpoint  = torch.load(MODEL_PATH, map_location=device)
CLASS_NAMES = checkpoint["class_names"]
NUM_SPECIES = len(CLASS_NAMES)

def load_model():
    m = models.efficientnet_v2_s(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_SPECIES)
    m.load_state_dict(checkpoint["model_state_dict"])
    m = m.to(device)
    m.eval()
    return m

model = load_model()
print(f"✅ Model loaded — {NUM_SPECIES} species\n")

# ── Provenance gate ──────────────────────────────────────────
# Checkpoints from train.py record the sha256 of the split manifests they were
# trained on. One without that record cannot be evaluated honestly: the
# original checkpoint scores ~95% even on official-CUB-test images that were
# never held out, well above the ~86-88% realistic ceiling, which means it has
# effectively seen the whole dataset. Every number below is then inflated by
# memorisation, so refuse to print one silently.
MODEL_IS_AUDITED = "split_manifest_sha256" in checkpoint
CONTAMINATION_WARNING = (
    "\n" + "!" * 62 + "\n"
    "  UNTRUSTWORTHY: this checkpoint records no training split.\n"
    "  Every accuracy figure below is inflated by memorisation and\n"
    "  must not be reported. Retrain with train.py for a real number.\n"
    + "!" * 62 + "\n"
)
if not MODEL_IS_AUDITED:
    print(CONTAMINATION_WARNING)

# ── Transform (same as training) ─────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Load test dataset ────────────────────────────────────────
print("Loading test dataset...")
test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)
print(f"   Test images: {len(test_dataset)}")
print(f"   Test species: {len(test_dataset.classes)}\n")


# ════════════════════════════════════════════════════════════
# FEATURE 1 — FULL EVALUATION METRICS
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("FEATURE 1 — Running Full Evaluation Metrics...")
print("=" * 60)

all_preds   = []
all_labels  = []
all_probs   = []
all_top5    = []

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        probs   = F.softmax(outputs, dim=1)

        top5_preds  = probs.topk(5, dim=1).indices.cpu().numpy()
        top1_preds  = probs.argmax(dim=1).cpu().numpy()

        all_preds.extend(top1_preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_top5.extend(top5_preds)

        if (i + 1) % 10 == 0:
            print(f"   Processed {(i+1)*BATCH_SIZE} images...")

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# Top-1 and Top-5 accuracy
top1_acc = (all_preds == all_labels).mean() * 100
top5_acc = np.mean([
    all_labels[i] in all_top5[i] for i in range(len(all_labels))
]) * 100

print(f"\n✅ Top-1 Accuracy : {top1_acc:.2f}%")
print(f"✅ Top-5 Accuracy : {top5_acc:.2f}%")

# Per-class precision, recall, F1
short_names = [n.split(".")[-1].replace("_", " ") for n in CLASS_NAMES]
report = classification_report(
    all_labels, all_preds,
    target_names=short_names,
    digits=3
)

# Save full report to text file
report_path = os.path.join(SAVE_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"Bird Species Identification — Evaluation Report\n")
    f.write(f"{'='*60}\n")
    f.write(f"Top-1 Accuracy : {top1_acc:.2f}%\n")
    f.write(f"Top-5 Accuracy : {top5_acc:.2f}%\n")
    f.write(f"{'='*60}\n\n")
    f.write(report)
print(f"\n📄 Full classification report saved to: {report_path}")

# ── Confusion Matrix (top 20 most confused species) ──────────
print("\nGenerating confusion matrix...")
prec, rec, f1, sup = precision_recall_fscore_support(
    all_labels, all_preds, average=None
)

# Find 20 species with lowest F1 (most confused)
worst20_idx  = np.argsort(f1)[:20]
worst20_names = [short_names[i][:18] for i in worst20_idx]

cm_full  = confusion_matrix(all_labels, all_preds)
cm_worst = cm_full[np.ix_(worst20_idx, worst20_idx)]

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
    cm_worst, annot=True, fmt="d", cmap="Blues",
    xticklabels=worst20_names,
    yticklabels=worst20_names,
    ax=ax, linewidths=0.5
)
ax.set_title("Confusion Matrix — 20 Most Confused Species", fontsize=14, pad=15)
ax.set_xlabel("Predicted Species", fontsize=12)
ax.set_ylabel("True Species", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
cm_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Confusion matrix saved to: {cm_path}")

# ── F1 Score bar chart (top 10 best + worst) ─────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

best10_idx   = np.argsort(f1)[-10:][::-1]
worst10_idx2 = np.argsort(f1)[:10]

ax1.barh([short_names[i][:25] for i in best10_idx],
         [f1[i]*100 for i in best10_idx], color="#1D9E75")
ax1.set_title("Top 10 Best Recognised Species (F1)", fontsize=12)
ax1.set_xlabel("F1 Score (%)")
ax1.set_xlim(0, 105)
for i, v in enumerate([f1[j]*100 for j in best10_idx]):
    ax1.text(v+1, i, f"{v:.1f}%", va="center", fontsize=9)

ax2.barh([short_names[i][:25] for i in worst10_idx2],
         [f1[i]*100 for i in worst10_idx2], color="#A32D2D")
ax2.set_title("Top 10 Hardest Species (F1)", fontsize=12)
ax2.set_xlabel("F1 Score (%)")
ax2.set_xlim(0, 105)
for i, v in enumerate([f1[j]*100 for j in worst10_idx2]):
    ax2.text(v+1, i, f"{v:.1f}%", va="center", fontsize=9)

plt.suptitle("Per-Species F1 Score Analysis", fontsize=14, y=1.01)
plt.tight_layout()
f1_path = os.path.join(SAVE_DIR, "f1_per_species.png")
plt.savefig(f1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 F1 score chart saved to: {f1_path}")

print("\n✅ FEATURE 1 COMPLETE\n")


# ════════════════════════════════════════════════════════════
# FEATURE 2 — MODEL COMPARISON TABLE
# ════════════════════════════════════════════════════════════
# Every number in this section is MEASURED from a real evaluation run.
# There are deliberately no placeholder rows: a model appears here only
# after it has actually been trained and evaluated on this test set.
# To add a row, train the model, run this script against it, and the
# entry is appended to results.json automatically.
print("=" * 60)
print("FEATURE 2 — Model Comparison")
print("=" * 60)

RESULTS_PATH = os.path.join(SAVE_DIR, "results.json")

# ── Measure this model's real footprint (not hardcoded) ──────
n_params  = sum(p.numel() for p in model.parameters())
state_MB  = sum(t.numel() * t.element_size()
                for t in model.state_dict().values()) / (1024 ** 2)

this_run = {
    "top1":        round(top1_acc, 2),
    "top5":        round(top5_acc, 2),
    "params_M":    round(n_params / 1e6, 2),
    "size_MB":     round(state_MB, 1),
    "split":       "custom 70/15/15 re-split (NOT the official CUB split)",
    "n_test":      int(len(all_labels)),
    "eval_date":   __import__("datetime").date.today().isoformat(),
}

# ── Accumulate across runs instead of inventing rows ─────────
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
        results = json.load(f)
else:
    results = {}
results["EfficientNetV2-S"] = this_run

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"📄 Measured results written to: {RESULTS_PATH}")

# ── Render the comparison only when there is something to compare ──
model_names = list(results.keys())

if len(model_names) < 2:
    print("\nOnly one model has been evaluated so far, so there is no")
    print("comparison chart to draw. Train and evaluate another backbone")
    print("(ResNet-50, MobileNetV3-Large, ...) and it will appear here.")
else:
    top1_scores = [results[m]["top1"]     for m in model_names]
    sizes       = [results[m]["size_MB"]  for m in model_names]
    params      = [results[m]["params_M"] for m in model_names]

    best = max(model_names, key=lambda m: results[m]["top1"])
    colors = ["#534AB7" if m == best else "#B4B2A9" for m in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, vals, title, unit in (
        (axes[0], top1_scores, "Top-1 Accuracy (%)", "%"),
        (axes[1], sizes,       "Model Size (MB)",    "MB"),
        (axes[2], params,      "Parameters (M)",     "M"),
    ):
        ax.bar(model_names, vals, color=colors)
        ax.set_title(title, fontsize=12)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v}{unit}", ha="center", va="bottom", fontsize=10)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylim(0, 100)

    legend_patch = mpatches.Patch(color="#534AB7", label=f"Best: {best}")
    fig.legend(handles=[legend_patch], loc="upper right", fontsize=10)
    plt.suptitle("Model Comparison — measured on the same test split", fontsize=14)
    plt.tight_layout()
    comp_path = os.path.join(SAVE_DIR, "model_comparison.png")
    plt.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Model comparison chart saved to: {comp_path}")

# ── Text table (measured rows only) ──────────────────────────
table_path = os.path.join(SAVE_DIR, "model_comparison_table.txt")
with open(table_path, "w") as f:
    f.write("Model Comparison Table — CUB-200-2011 Bird Dataset\n")
    f.write("All figures measured; no placeholder rows.\n")
    f.write(f"Split: {this_run['split']}\n")
    f.write("=" * 62 + "\n")
    f.write(f"{'Model':<25} {'Top-1':>8} {'Top-5':>8} {'Params M':>10} {'Size MB':>9}\n")
    f.write("-" * 62 + "\n")
    for m in model_names:
        r = results[m]
        f.write(f"{m:<25} {r['top1']:>7.1f}% {r['top5']:>7.1f}% "
                f"{r['params_M']:>10.1f} {r['size_MB']:>8.1f}\n")
    f.write("=" * 62 + "\n")
print(f"📄 Model comparison table saved to: {table_path}")

print("\n✅ FEATURE 2 COMPLETE\n")


# ════════════════════════════════════════════════════════════
# FEATURE 3 — GRAD-CAM VISUALISATION
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("FEATURE 3 — Grad-CAM Visualisation")
print("=" * 60)

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # Target the last conv layer of EfficientNetV2-S
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # ── Run Grad-CAM on 6 random test images ─────────────────
    import random
    species_list = os.listdir(TEST_DIR)
    random.shuffle(species_list)
    sample_species = species_list[:6]

    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    fig.suptitle("Grad-CAM — What the model focuses on", fontsize=14)

    raw_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
    ])
    norm_transform = transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )

    for col, species in enumerate(sample_species):
        sp_path = os.path.join(TEST_DIR, species)
        img_file = os.listdir(sp_path)[0]
        img_path = os.path.join(sp_path, img_file)

        pil_img = Image.open(img_path).convert("RGB")
        raw_tensor  = raw_transform(pil_img)
        norm_tensor = norm_transform(raw_tensor).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            out   = model(norm_tensor)
            probs = F.softmax(out, dim=1)
            pred_idx  = probs.argmax().item()
            pred_conf = probs[0][pred_idx].item() * 100
            pred_name = CLASS_NAMES[pred_idx].split(".")[-1].replace("_", " ")

        # Generate Grad-CAM
        targets    = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=norm_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]

        rgb_img = raw_tensor.permute(1, 2, 0).numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Original image (top row)
        axes[0][col].imshow(rgb_img)
        axes[0][col].set_title(
            species.split(".")[-1].replace("_", " ")[:20],
            fontsize=8
        )
        axes[0][col].axis("off")

        # Grad-CAM overlay (bottom row)
        axes[1][col].imshow(cam_image)
        axes[1][col].set_title(
            f"Pred: {pred_name[:18]}\n{pred_conf:.1f}%",
            fontsize=8,
            color="#1D9E75" if pred_name in species else "#A32D2D"
        )
        axes[1][col].axis("off")

    axes[0][0].set_ylabel("Original", fontsize=10)
    axes[1][0].set_ylabel("Grad-CAM", fontsize=10)

    plt.tight_layout()
    cam_path = os.path.join(SAVE_DIR, "gradcam_visualisation.png")
    plt.savefig(cam_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Grad-CAM visualisation saved to: {cam_path}")
    print("\n✅ FEATURE 3 COMPLETE\n")

except ImportError:
    print("⚠️  pytorch-grad-cam not installed.")
    print("   Run: pip3 install grad-cam")
    print("   Then run this script again.\n")


# ════════════════════════════════════════════════════════════
# FEATURE 4 — CONFIDENCE CALIBRATION
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("FEATURE 4 — Confidence Calibration")
print("=" * 60)

# ── Collect confidence scores and correctness ─────────────────
confidences = np.max(all_probs, axis=1)   # highest prob for each image
correctness = (all_preds == all_labels).astype(float)

# ── Bin into 10 buckets (0-10%, 10-20%, ... 90-100%) ─────────
n_bins   = 10
bin_size = 1.0 / n_bins
bins     = np.arange(0, 1 + bin_size, bin_size)

bin_accs  = []
bin_confs = []
bin_sizes = []

for i in range(n_bins):
    lo, hi = bins[i], bins[i + 1]
    mask   = (confidences > lo) & (confidences <= hi)
    if mask.sum() > 0:
        bin_accs.append(correctness[mask].mean())
        bin_confs.append(confidences[mask].mean())
        bin_sizes.append(mask.sum())
    else:
        bin_accs.append(0)
        bin_confs.append((lo + hi) / 2)
        bin_sizes.append(0)

bin_accs  = np.array(bin_accs)
bin_confs = np.array(bin_confs)

# ── ECE (Expected Calibration Error) ─────────────────────────
n        = len(confidences)
bin_weights = np.array(bin_sizes) / n
ece      = float(np.sum(bin_weights * np.abs(bin_accs - bin_confs)) * 100)

print(f"   Expected Calibration Error (ECE): {ece:.2f}%")
print(f"   (Lower is better — 0% = perfect calibration)")

# ── Plot reliability diagram ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Reliability diagram
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(n_bins)]
ax1.bar(bin_centers, bin_accs, width=bin_size * 0.9,
        alpha=0.7, color="#534AB7", label="Actual accuracy")
ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
ax1.fill_between([0, 1], [0, 1], alpha=0.1, color="gray")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel("Confidence Score", fontsize=12)
ax1.set_ylabel("Actual Accuracy", fontsize=12)
ax1.set_title(f"Reliability Diagram\nECE = {ece:.2f}%", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Confidence histogram
ax2.hist(confidences, bins=20, color="#1D9E75", edgecolor="white", linewidth=0.5)
ax2.set_xlabel("Confidence Score", fontsize=12)
ax2.set_ylabel("Number of Predictions", fontsize=12)
ax2.set_title("Distribution of Confidence Scores", fontsize=12)
ax2.axvline(x=0.6, color="#A32D2D", linestyle="--",
            linewidth=1.5, label="60% threshold")
ax2.axvline(x=confidences.mean(), color="#BA7517", linestyle="--",
            linewidth=1.5, label=f"Mean ({confidences.mean()*100:.1f}%)")
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.suptitle("Confidence Calibration Analysis", fontsize=14)
plt.tight_layout()
calib_path = os.path.join(SAVE_DIR, "confidence_calibration.png")
plt.savefig(calib_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Calibration diagram saved to: {calib_path}")

# ── Save calibration numbers to text ─────────────────────────
calib_txt = os.path.join(SAVE_DIR, "calibration_report.txt")
with open(calib_txt, "w") as f:
    f.write("Confidence Calibration Report\n")
    f.write("="*50 + "\n")
    f.write(f"Expected Calibration Error (ECE): {ece:.2f}%\n")
    f.write(f"Mean confidence                 : {confidences.mean()*100:.2f}%\n")
    f.write(f"Mean accuracy                   : {top1_acc:.2f}%\n\n")
    f.write(f"{'Confidence Bin':<20} {'Actual Accuracy':>16} {'# Images':>10}\n")
    f.write("-"*50 + "\n")
    for i in range(n_bins):
        lo = bins[i] * 100
        hi = bins[i+1] * 100
        f.write(f"{lo:.0f}% – {hi:.0f}%{'':<12}"
                f"{bin_accs[i]*100:>15.1f}%"
                f"{bin_sizes[i]:>10}\n")
    f.write("="*50 + "\n")
print(f"📄 Calibration report saved to: {calib_txt}")

print("\n✅ FEATURE 4 COMPLETE\n")


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("ALL DONE! Files saved to:", SAVE_DIR)
print("=" * 60)
print(f"""
Files generated:
  📄 classification_report.txt   ← Precision, Recall, F1 per species
  📊 confusion_matrix.png        ← Which species get confused
  📊 f1_per_species.png          ← Best and worst species
  📊 model_comparison.png        ← EfficientNetV2-S vs others
  📄 model_comparison_table.txt  ← Numbers for your paper table
  📊 gradcam_visualisation.png   ← What the model looks at
  📊 confidence_calibration.png  ← Reliability diagram
  📄 calibration_report.txt      ← ECE and calibration numbers

Key numbers for your paper:
  Top-1 Accuracy : {top1_acc:.2f}%
  Top-5 Accuracy : {top5_acc:.2f}%
  ECE            : {ece:.2f}%
""")
