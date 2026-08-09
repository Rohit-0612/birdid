"""Offline evaluation: test-set metrics and confidence calibration.

Both functions were written long before there was anywhere to show them and
were never wired to a UI element. They are kept here — out of bird_core, which
the API imports on every request — because they pull in sklearn/seaborn, walk a
full ImageFolder and take minutes to run. Call them from a script or the Gradio
Evaluation tab, never from a request handler.

Every result is prefixed with the contamination warning while the loaded
checkpoint has no recorded training split, because a number from a
leak-contaminated model is worse than no number at all.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader

from bird_core import (
    CLASS_NAMES,
    CONTAMINATION_WARNING,
    MODEL_IS_AUDITED,
    PROJECT_DIR,
    SAVE_DIR,
    TEST_DIR,
    device,
    model,
    val_transform,
)

# ════════════════════════════════════════════════════════════
# EVALUATION METRICS FUNCTION
# ════════════════════════════════════════════════════════════
def run_evaluation():
    if not os.path.exists(TEST_DIR):
        return (
            "❌ Test folder not found!\n\n"
            f"Expected at: {TEST_DIR}\n\n"
            "Please download the test images from Kaggle or Colab first.\n"
            "See instructions in the README."
        )
    try:
        from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
        import seaborn as sns

        print("Loading test dataset...")
        test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)
        test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

        all_preds, all_labels, all_probs, all_top5 = [], [], [], []

        print("Running evaluation...")
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images  = images.to(device)
                outputs = model(images)
                probs   = F.softmax(outputs, dim=1)
                top5    = probs.topk(5, dim=1).indices.cpu().numpy()
                preds   = probs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                all_top5.extend(top5)
                if (i+1) % 5 == 0:
                    print(f"   Batch {i+1}/{len(test_loader)}...")

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs  = np.array(all_probs)

        top1_acc = (all_preds == all_labels).mean() * 100
        top5_acc = np.mean([all_labels[i] in all_top5[i] for i in range(len(all_labels))]) * 100

        short_names = [n.split(".")[-1].replace("_", " ") for n in CLASS_NAMES]
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

        best10  = np.argsort(f1)[-10:][::-1]
        worst10 = np.argsort(f1)[:10]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.barh([short_names[i][:25] for i in best10],  [f1[i]*100 for i in best10],  color="#1D9E75")
        ax1.set_title("Top 10 Best Recognised Species", fontsize=12)
        ax1.set_xlabel("F1 Score (%)")
        for i, v in enumerate([f1[j]*100 for j in best10]):
            ax1.text(v+0.5, i, f"{v:.1f}%", va="center", fontsize=9)

        ax2.barh([short_names[i][:25] for i in worst10], [f1[i]*100 for i in worst10], color="#A32D2D")
        ax2.set_title("Top 10 Hardest Species", fontsize=12)
        ax2.set_xlabel("F1 Score (%)")
        for i, v in enumerate([f1[j]*100 for j in worst10]):
            ax2.text(v+0.5, i, f"{v:.1f}%", va="center", fontsize=9)

        plt.suptitle("Per-Species F1 Score Analysis", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "f1_per_species.png"), dpi=150, bbox_inches="tight")
        plt.close()

        report = classification_report(all_labels, all_preds, target_names=short_names, digits=3)
        report_path = os.path.join(SAVE_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\nTop-5 Accuracy: {top5_acc:.2f}%\n\n{report}")

        return (
            ("" if MODEL_IS_AUDITED else CONTAMINATION_WARNING) +
            f"✅ EVALUATION COMPLETE!\n\n"
            f"Split          : {os.path.relpath(TEST_DIR, PROJECT_DIR)}\n"
            f"Top-1 Accuracy : {top1_acc:.2f}%\n"
            f"Top-5 Accuracy : {top5_acc:.2f}%\n\n"
            f"Best species   : {short_names[best10[0]]} (F1: {f1[best10[0]]*100:.1f}%)\n"
            f"Hardest species: {short_names[worst10[0]]} (F1: {f1[worst10[0]]*100:.1f}%)\n\n"
            f"Files saved to: {SAVE_DIR}\n"
            f"  - classification_report.txt\n"
            f"  - f1_per_species.png"
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ════════════════════════════════════════════════════════════
# CONFIDENCE CALIBRATION FUNCTION
# ════════════════════════════════════════════════════════════
def run_calibration():
    if not os.path.exists(TEST_DIR):
        return "❌ Test folder not found! See instructions above."
    try:
        test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)
        test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images  = images.to(device)
                outputs = model(images)
                probs   = F.softmax(outputs, dim=1)
                all_preds.extend(probs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs  = np.array(all_probs)
        confidences = np.max(all_probs, axis=1)
        correctness = (all_preds == all_labels).astype(float)

        n_bins = 10
        bins   = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs, bin_sizes = [], [], []

        for i in range(n_bins):
            mask = (confidences > bins[i]) & (confidences <= bins[i+1])
            if mask.sum() > 0:
                bin_accs.append(correctness[mask].mean())
                bin_confs.append(confidences[mask].mean())
                bin_sizes.append(int(mask.sum()))
            else:
                bin_accs.append(0)
                bin_confs.append((bins[i]+bins[i+1])/2)
                bin_sizes.append(0)

        bin_accs  = np.array(bin_accs)
        ece = float(np.sum((np.array(bin_sizes)/len(confidences)) * np.abs(bin_accs - np.array(bin_confs))) * 100)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        bin_centers = [(bins[i]+bins[i+1])/2 for i in range(n_bins)]
        ax1.bar(bin_centers, bin_accs, width=0.09, alpha=0.75, color="#534AB7", label="Actual accuracy")
        ax1.plot([0,1],[0,1],"k--", linewidth=1.5, label="Perfect calibration")
        ax1.set_xlabel("Confidence Score"); ax1.set_ylabel("Actual Accuracy")
        ax1.set_title(f"Reliability Diagram\nECE = {ece:.2f}%")
        ax1.set_xlim(0,1); ax1.set_ylim(0,1)
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.hist(confidences, bins=20, color="#1D9E75", edgecolor="white")
        ax2.axvline(x=0.6, color="#A32D2D", linestyle="--", label="60% threshold")
        ax2.axvline(x=confidences.mean(), color="#BA7517", linestyle="--",
                    label=f"Mean ({confidences.mean()*100:.1f}%)")
        ax2.set_xlabel("Confidence Score"); ax2.set_ylabel("Number of Predictions")
        ax2.set_title("Distribution of Confidence Scores")
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.suptitle("Confidence Calibration Analysis", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "confidence_calibration.png"), dpi=150, bbox_inches="tight")
        plt.close()

        return (
            ("" if MODEL_IS_AUDITED else CONTAMINATION_WARNING) +
            f"✅ CALIBRATION COMPLETE!\n\n"
            f"ECE (Expected Calibration Error) : {ece:.2f}%\n"
            f"Mean confidence                  : {confidences.mean()*100:.2f}%\n"
            f"Mean accuracy                    : {(all_preds==all_labels).mean()*100:.2f}%\n\n"
            f"ECE closer to 0% = better calibrated model\n\n"
            f"Saved to: {SAVE_DIR}/confidence_calibration.png"
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"
