"""
Evaluates the fine-tuned Swin Transformer V2-B on the full HAM10000 held-out test cohort (N=4,500).
Computes:
- Per-class Precision, Sensitivity (Recall), Specificity, F1-Score, OvR ROC-AUC, Support (N)
- Macro-average and Weighted-average metrics with rigorous mathematical consistency
- Balanced Accuracy Score
- Expected Calibration Error (ECE)
- Saves outputs to outputs/tables/table_internal_ham10000_exact.tex and .csv
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score, confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.models.swin_classifier import EnhancedSwinTransformer

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}


class TestHAMDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_p = self.img_dir / f"{row['image_id']}.jpg"
        img = Image.open(img_p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = LABEL_MAP[row["dx"]]
        return img, label


def compute_ece(probs, labels, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    test_csv = Path("data/testing-set/ham10000_test_dataset_4500.csv")
    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    df = pd.read_csv(test_csv)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TestHAMDataset(df, img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    weights_path = "weights/swin_vit_xai_model.pth"
    model = EnhancedSwinTransformer(num_classes=7, use_batchnorm=True, checkpoint_path=weights_path).to(device)
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    t0 = time.time()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(torch.argmax(probs, dim=-1).cpu().numpy())
            all_labels.append(targets.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    elapsed = time.time() - t0
    print(f"Evaluated {len(all_labels)} images in {elapsed:.2f}s ({elapsed/len(all_labels)*1000:.2f} ms/image).")

    # Metrics computation
    acc = np.mean(all_preds == all_labels)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    ece = compute_ece(all_probs, all_labels)

    # One-hot labels for AUC
    one_hot = np.eye(7)[all_labels]
    ovr_aucs = roc_auc_score(one_hot, all_probs, multi_class="ovr", average=None)
    macro_auc = np.mean(ovr_aucs)
    weighted_auc = roc_auc_score(one_hot, all_probs, multi_class="ovr", average="weighted")

    # Confusion matrix for per-class specificity
    cm = confusion_matrix(all_labels, all_preds)
    specificities = []
    for i in range(7):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm[:, i], i))
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    # Classification report
    rep = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)

    rows = []
    for i, c in enumerate(CLASSES):
        rows.append({
            "Class": c.upper(),
            "Precision": rep[c]["precision"],
            "Sensitivity (Recall)": rep[c]["recall"],
            "Specificity": specificities[i],
            "F1-Score": rep[c]["f1-score"],
            "OvR ROC-AUC": ovr_aucs[i],
            "Support (N)": int(rep[c]["support"])
        })

    # Exact macro and weighted arithmetic averages
    macro_p = np.mean([r["Precision"] for r in rows])
    macro_r = np.mean([r["Sensitivity (Recall)"] for r in rows])
    macro_s = np.mean([r["Specificity"] for r in rows])
    macro_f1 = np.mean([r["F1-Score"] for r in rows])

    rows.append({
        "Class": "MACRO AVERAGE",
        "Precision": macro_p,
        "Sensitivity (Recall)": macro_r,
        "Specificity": macro_s,
        "F1-Score": macro_f1,
        "OvR ROC-AUC": macro_auc,
        "Support (N)": len(all_labels)
    })

    weighted_p = rep["weighted avg"]["precision"]
    weighted_r = rep["weighted avg"]["recall"]
    weighted_f1 = rep["weighted avg"]["f1-score"]

    rows.append({
        "Class": "WEIGHTED AVERAGE",
        "Precision": weighted_p,
        "Sensitivity (Recall)": weighted_r,
        "Specificity": np.sum([s * rep[c]["support"] for s, c in zip(specificities, CLASSES)]) / len(all_labels),
        "F1-Score": weighted_f1,
        "OvR ROC-AUC": weighted_auc,
        "Support (N)": len(all_labels)
    })

    res_df = pd.DataFrame(rows)
    print("\n=== EXACT HAM10000 TEST METRICS (N=4,500) ===")
    print(res_df.to_string(index=False))
    print(f"\nOverall Top-1 Accuracy: {acc*100:.2f}%")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"Expected Calibration Error (ECE): {ece*100:.2f}%\n")

    res_df.to_csv("outputs/tables/table_internal_ham10000_exact.csv", index=False)

    # Format LaTeX
    tex_str = res_df.to_latex(index=False, float_format="%.4f")
    with open("outputs/tables/table_internal_ham10000_exact.tex", "w") as f:
        f.write(tex_str)
    print("Saved exact Table 1 to outputs/tables/table_internal_ham10000_exact.tex")


if __name__ == "__main__":
    main()
