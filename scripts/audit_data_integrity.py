"""
P0 Audit 1: Data & Checkpoint Integrity Check.
Verifies:
  1. Unsegmented (ISIC 2019 extracted) vs Segmented Cutout (segmented-ham10000) accuracy on the 4,500 test split.
  2. Exact class distribution, Tschandl mask availability, and lesion_id cluster counts.
  3. Whether the saved checkpoint in weights/swin_vit_xai_model.pth is epoch 5 (best) or epoch 10 (last).
  4. External cohort composition in ISIC 2019 (pre-2018 archive vs HAM10000 vs BCN_20000) and aspect/vignette properties.
  5. Confidence saturation and predicted class distribution skew.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from src.models.swin_classifier import EnhancedSwinTransformer

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


class ImgDS(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths = paths
        self.labels = labels
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), self.labels[idx]


@torch.no_grad()
def eval_folder(model, df, folder, label_map, device, batch_size=128):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    paths, labels = [], []
    for _, r in df.iterrows():
        p = Path(folder) / f"{r['image_id']}.jpg"
        if p.exists():
            paths.append(p)
            labels.append(label_map[r["dx"]])
    if not paths:
        return None
    dl = DataLoader(ImgDS(paths, labels, tf), batch_size=batch_size, num_workers=0)
    probs_all = []
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            probs = F.softmax(model(x).float(), dim=1)
        probs_all.append(probs.cpu())
    probs_np = torch.cat(probs_all).numpy()
    y_true = np.array(labels)
    y_pred = probs_np.argmax(1)
    pred_counts = {CLASSES[i]: int((y_pred == i).sum()) for i in range(7)}
    return {
        "n_evaluated": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mean_max_confidence": float(probs_np.max(1).mean()),
        "median_max_confidence": float(np.median(probs_np.max(1))),
        "predicted_class_counts": pred_counts,
        "per_class_recall": {CLASSES[i]: float((y_pred[y_true == i] == i).mean()) for i in range(7)},
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = "weights/swin_vit_xai_model.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    label_map = ckpt["label_map"]
    hist = ckpt.get("training_history", {})

    print("=== 1. CHECKPOINT & TRAINING HISTORY ===")
    print("Label map:", label_map)
    print("Model config:", ckpt.get("model_config"))
    val_accs = hist.get("val_accuracies", [])
    val_losses = hist.get("val_losses", [])
    if val_accs:
        best_ep = int(np.argmax(val_accs)) + 1
        print(f"Epochs recorded: {len(val_accs)}, Best val_acc epoch: {best_ep} ({max(val_accs):.4f}), Final epoch ({len(val_accs)}) val_acc: {val_accs[-1]:.4f}, val_loss: {val_losses[-1]:.4f} (min val_loss: {min(val_losses):.4f} at ep {int(np.argmin(val_losses))+1})")

    model = EnhancedSwinTransformer(num_classes=7, use_batchnorm=True, checkpoint_path=ckpt_path).to(device).eval()

    df_test = pd.read_csv("data/testing-set/ham10000_test_dataset_4500.csv")
    mask_dir = Path("data/ham10000_masks/HAM10000_segmentations_lesion_tschandl")
    mask_exists = sum((mask_dir / f"{img_id}_segmentation.png").exists() for img_id in df_test["image_id"])

    print("\n=== 2. TEST COHORT STRUCTURE ===")
    print(f"Rows: {len(df_test)}, Unique lesions: {df_test['lesion_id'].nunique()}, Tschandl masks present: {mask_exists}/{len(df_test)}")

    seg_dir = Path("data/segmented-ham10000/segmented_images")
    seg_files = list(seg_dir.glob("*.jpg"))
    oversampled_files = [p.name for p in seg_files if p.stem.startswith("oversampled_")]
    base_seg_ids = {p.stem.replace("oversampled_", "") for p in seg_files}
    test_ids = set(df_test["image_id"])
    print(f"segmented-ham10000 total files: {len(seg_files)}, oversampled_* files: {len(oversampled_files)}, overlap with test_ids: {len(base_seg_ids & test_ids)}/{len(test_ids)}")

    # Compare evaluation on unsegmented vs segmented cutouts
    print("\n=== 3. UNSEGMENTED VS BLACK-CUTOUT EVALUATION (N=4,500) ===")
    res_unseg = eval_folder(model, df_test, "data/external_isic2019/extracted/train-image/image", label_map, device)
    res_seg = eval_folder(model, df_test, "data/segmented-ham10000/segmented_images", label_map, device)
    print("Unsegmented (ISIC 2019 HAM):", json.dumps(res_unseg, indent=2))
    print("Pre-segmented Cutouts (segmented-ham10000):", json.dumps(res_seg, indent=2))

    # 4. Inspect ISIC 2019 composition
    print("\n=== 4. ISIC 2019 COHORT BREAKDOWN ===")
    gt = pd.read_csv("data/external_isic2019/ISIC_2019_Training_GroundTruth.csv")
    gt["num"] = gt["image"].apply(lambda s: int(s.replace("ISIC_", "").split("_")[0]))
    pre_ham = gt[gt["num"] < 24306]
    ham_range = gt[(gt["num"] >= 24306) & (gt["num"] <= 34320)]
    mid_range = gt[(gt["num"] > 34320) & (gt["num"] < 50000)]
    bcn_range = gt[gt["num"] >= 50000]
    print(f"pre-2018 (<24306): {len(pre_ham)}, HAM range (24306..34320): {len(ham_range)}, mid (34321..49999): {len(mid_range)}, BCN (>=50000): {len(bcn_range)}")
    print("UNK/SCC counts in pre-2018:", int(pre_ham["SCC"].sum()), "UNK:", int(pre_ham["UNK"].sum()))
    print("UNK/SCC counts in BCN (>=50000):", int(bcn_range["SCC"].sum()), "UNK:", int(bcn_range["UNK"].sum()))

    # Image sizes and dark vignette rate check (sample 100 from each cohort)
    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    for cname, sub in [("HAM_test", df_test.rename(columns={"image_id": "image"}).head(100)), ("pre2018_archive", pre_ham.head(100)), ("BCN_20000", bcn_range.head(100))]:
        shapes = set()
        vig_count = 0
        for im_id in sub["image"]:
            p = img_dir / f"{im_id}.jpg"
            if not p.exists():
                continue
            im = cv2.imread(str(p))
            shapes.add(im.shape[:2])
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            corners = np.array([gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]])
            if (corners < 25).all():
                vig_count += 1
        print(f"Cohort {cname}: sample shapes={shapes}, dark-corner vignette rate={vig_count}/100")

    out_report = {
        "checkpoint": ckpt_path,
        "training_history": hist,
        "test_rows": len(df_test),
        "test_unique_lesions": int(df_test["lesion_id"].nunique()),
        "test_tschandl_masks_present": int(mask_exists),
        "unsegmented_eval": res_unseg,
        "cutout_eval": res_seg,
        "isic2019_ranges": {
            "pre_2018_lt_24306": len(pre_ham),
            "ham_24306_to_34320": len(ham_range),
            "mid_34321_to_49999": len(mid_range),
            "bcn_ge_50000": len(bcn_range),
        },
    }
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    with open("outputs/tables/audit_data_integrity.json", "w") as f:
        json.dump(out_report, f, indent=2)
    print("\nSaved audit report to outputs/tables/audit_data_integrity.json")


if __name__ == "__main__":
    main()
