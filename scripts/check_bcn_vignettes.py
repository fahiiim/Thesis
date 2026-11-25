import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
from src.xai.csm import ClinicalSaliencyMask
from src.xai.csm_v2 import ClinicalSaliencyMaskV2
from src.utils.data_v2 import load_rgb, preprocess


def main():
    gt = pd.read_csv("data/external_isic2019/ISIC_2019_Training_GroundTruth.csv")
    gt["num"] = gt["image"].apply(lambda s: int(s.replace("ISIC_", "").split("_")[0]))
    bcn = gt[gt["num"] >= 50000].sample(n=200, random_state=42)
    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    csm1, csm2 = ClinicalSaliencyMask(), ClinicalSaliencyMaskV2()

    vig_tot, fail_v1_area, fail_v2_area, touch_v1, touch_v2 = 0, 0, 0, 0, 0
    mean_area_v1, mean_area_v2 = [], []
    for _, r in bcn.iterrows():
        p = img_dir / f"{r['image']}.jpg"
        if not p.exists():
            continue
        rgb = load_rgb(p)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        has_vig = (np.array([gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]) < 25).all()
        if not has_vig:
            continue
        vig_tot += 1
        _, comp1 = csm1.generate(preprocess(rgb, 224, "legacy"))
        m1 = (comp1["lesion_mask"] > 0).astype(np.uint8)
        res2 = csm2.generate(preprocess(rgb, 224, "parity"))
        m2 = (res2.lesion_mask > 0).astype(np.uint8)
        mean_area_v1.append(m1.mean())
        mean_area_v2.append(m2.mean())
        if m1.mean() > 0.70:
            fail_v1_area += 1
        if m2.mean() > 0.70:
            fail_v2_area += 1
        if m1[0].mean() > 0.5 or m1[-1].mean() > 0.5:
            touch_v1 += 1
        if m2[0].mean() > 0.5 or m2[-1].mean() > 0.5:
            touch_v2 += 1

    print(f"BCN vignette images checked: {vig_tot}/200")
    print(f"  CSM-v1 >70% frame flood: {fail_v1_area}/{vig_tot} ({fail_v1_area/vig_tot*100:.1f}%), border flood: {touch_v1}/{vig_tot}, mean fg={np.mean(mean_area_v1)*100:.1f}%")
    print(f"  CSM-v2 >70% frame flood: {fail_v2_area}/{vig_tot} ({fail_v2_area/vig_tot*100:.1f}%), border flood: {touch_v2}/{vig_tot}, mean fg={np.mean(mean_area_v2)*100:.1f}%")


if __name__ == "__main__":
    main()
