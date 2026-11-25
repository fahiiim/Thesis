"""
Clinical Saliency Mask v2 (CSM-v2): Stolz-aligned dermoscopic ABCD prior without deep segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

EPS = 1e-8

STOLZ_COLOURS_LAB: dict[str, tuple[float, float, float]] = {
    "white": (88.0, 0.0, 5.0),
    "red": (50.0, 45.0, 20.0),
    "light_brown": (60.0, 14.0, 28.0),
    "dark_brown": (35.0, 14.0, 20.0),
    "blue_gray": (45.0, -3.0, -12.0),
    "black": (15.0, 2.0, 2.0),
}


@dataclass
class CSMConfig:
    dark_fov_threshold: int = 25
    hair_kernel: int = 9
    hair_threshold: int = 12
    flatfield_sigma_frac: float = 0.25
    grabcut_iters: int = 4
    min_lesion_frac: float = 0.01
    asym_shape_tau: float = 0.12
    asym_colour_tau: float = 0.25
    border_abrupt_tau: float = 2.5
    colour_max_delta_e: float = 22.0
    colour_min_frac: float = 0.02
    dot_sigmas: tuple[float, ...] = (1.0, 1.5)
    globule_sigmas: tuple[float, ...] = (2.5, 3.5)
    blob_tau: float = 6.0
    min_dots: int = 3
    min_globules: int = 2
    ridge_sigmas: tuple[float, ...] = (1.0, 1.5)
    ridge_tau: float = 3.0
    network_min_frac: float = 0.06
    network_min_holes: int = 4
    streak_min_frac: float = 0.03
    structureless_tau: float = 4.0
    structureless_min_frac: float = 0.10
    tds_weights: tuple[float, float, float, float] = (1.3, 0.1, 0.5, 0.5)


@dataclass
class CSMResult:
    saliency: np.ndarray
    components: dict[str, np.ndarray]
    lesion_mask: np.ndarray
    fov_mask: np.ndarray
    quality: float
    scores: dict[str, float] = field(default_factory=dict)
    channel: str = ""


def _norm01(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    x = x.astype(np.float32)
    vals = x[mask > 0] if mask is not None and np.any(mask) else x
    lo, hi = float(vals.min()), float(vals.max())
    out = (x - lo) / (hi - lo + EPS)
    out = np.clip(out, 0.0, 1.0)
    return out * (mask > 0) if mask is not None else out


def _largest_component(mask: np.ndarray, score_fn=None) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(np.uint8)
    scores = [score_fn(labels == i, stats[i]) if score_fn else stats[i, cv2.CC_STAT_AREA] for i in range(1, n)]
    return (labels == (1 + int(np.argmax(scores)))).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    m = cv2.copyMakeBorder((mask > 0).astype(np.uint8) * 255, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = m.copy()
    cv2.floodFill(flood, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 255)
    filled = (m | cv2.bitwise_not(flood)) > 0
    return filled[1:-1, 1:-1].astype(np.uint8)


def _inner_ring(region: np.ndarray, width: int) -> np.ndarray:
    padded = cv2.copyMakeBorder(region.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((2 * width + 1, 2 * width + 1), np.uint8))[1:-1, 1:-1]
    return (region > 0) & (eroded == 0)


def _otsu_in_mask(channel: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    vals = channel[mask > 0].astype(np.uint8).reshape(-1, 1)
    if vals.size < 50 or vals.min() == vals.max():
        return 127.0, 0.0
    t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v = vals.astype(np.float32).ravel()
    lo, hi = v[v <= t], v[v > t]
    if lo.size == 0 or hi.size == 0:
        return float(t), 0.0
    w0, w1 = lo.size / v.size, hi.size / v.size
    eta = w0 * w1 * (lo.mean() - hi.mean()) ** 2 / (v.var() + EPS)
    return float(t), float(eta)


def _hessian_eigs(img: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = cv2.GaussianBlur(img, (0, 0), sigma)
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3) * sigma ** 2
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) * sigma ** 2
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) * sigma ** 2
    tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
    l1 = 0.5 * (dxx + dyy - tmp)
    l2 = 0.5 * (dxx + dyy + tmp)
    theta_line = 0.5 * np.arctan2(2 * dxy, dxx - dyy) + np.pi / 2
    return l1, l2, theta_line


class ClinicalSaliencyMaskV2:
    """Stolz-aligned clinical prior. Call ``generate(rgb_uint8)``."""

    def __init__(self, config: CSMConfig | None = None) -> None:
        self.cfg = config or CSMConfig()

    def fov_mask(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        dark = (gray < self.cfg.dark_fov_threshold).astype(np.uint8)
        n, labels = cv2.connectedComponents(dark, connectivity=8)
        border_labels = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
        outside = np.isin(labels, [lbl for lbl in border_labels if lbl != 0])
        fov = (~outside).astype(np.uint8)
        fov = cv2.erode(fov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        return fov if fov.mean() > 0.2 else np.ones_like(gray, np.uint8)

    def remove_hair(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.cfg.hair_kernel, self.cfg.hair_kernel))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
        hair = (blackhat > self.cfg.hair_threshold).astype(np.uint8)
        frac = hair.mean()
        if frac < 0.002 or frac > 0.25:
            return rgb
        hair = cv2.dilate(hair, np.ones((3, 3), np.uint8))
        return cv2.inpaint(rgb, hair, 3, cv2.INPAINT_TELEA)

    def flatfield_lab(self, rgb: np.ndarray, fov: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lum = lab[..., 0]
        sigma = self.cfg.flatfield_sigma_frac * min(rgb.shape[:2])
        f = fov.astype(np.float32)
        bg = cv2.GaussianBlur(lum * f, (0, 0), sigma) / (cv2.GaussianBlur(f, (0, 0), sigma) + EPS)
        lab[..., 0] = np.clip(lum / (bg + EPS) * float(lum[fov > 0].mean()), 0, 255)
        return lab

    def segment(self, rgb: np.ndarray, lab: np.ndarray, fov: np.ndarray) -> tuple[np.ndarray, float, str]:
        h, w = fov.shape
        candidates: dict[str, np.ndarray] = {
            "darkness": 255.0 - lab[..., 0],
            "redness_a": lab[..., 1],
            "blue_darkness": 255.0 - rgb[..., 2].astype(np.float32),
        }
        pix = lab[fov > 0].reshape(-1, 3)
        if pix.shape[0] > 10:
            mu = pix.mean(0)
            _, _, vt = np.linalg.svd(pix - mu, full_matrices=False)
            pc1 = ((lab.reshape(-1, 3) - mu) @ vt[0]).reshape(h, w)
            if float(np.std(pc1[fov > 0])) > 1e-6 and float(np.std(candidates["darkness"][fov > 0])) > 1e-6:
                if np.corrcoef(pc1[fov > 0], candidates["darkness"][fov > 0])[0, 1] < 0:
                    pc1 = -pc1
            candidates["pca1"] = pc1
        ring = _inner_ring(fov, 4)

        best: tuple[float, np.ndarray, str] = (-1.0, np.zeros_like(fov), "")
        for name, ch in candidates.items():
            ch8 = (_norm01(ch, fov) * 255).astype(np.uint8)
            t, eta = _otsu_in_mask(ch8, fov)
            hi = ((ch8 > t) & (fov > 0)).astype(np.uint8)
            lo = ((ch8 <= t) & (fov > 0)).astype(np.uint8)
            contact_hi = hi[ring].mean() if ring.any() else 0.0
            contact_lo = lo[ring].mean() if ring.any() else 1.0
            cand = hi if contact_hi <= contact_lo else lo
            if eta > best[0]:
                best = (eta, cand, name)
        eta, otsu, channel = best

        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        otsu = cv2.morphologyEx(cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k7), cv2.MORPH_CLOSE, k7)

        def score(comp: np.ndarray, stats: np.ndarray) -> float:
            contact = comp[ring].mean() if ring.any() else 0.0
            return float(stats[cv2.CC_STAT_AREA]) * (1.0 - contact)

        otsu = _fill_holes(_largest_component(otsu, score))
        if otsu.sum() < self.cfg.min_lesion_frac * fov.sum():
            return otsu, 0.0, channel

        refined = self._grabcut(rgb, otsu, fov)
        inter = float((refined & otsu).sum())
        union = float((refined | otsu).sum()) + EPS
        quality = inter / union
        return refined, quality, channel

    def _grabcut(self, rgb: np.ndarray, init: np.ndarray, fov: np.ndarray) -> np.ndarray:
        gc = np.full(init.shape, cv2.GC_PR_BGD, np.uint8)
        far = cv2.dilate(init, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) == 0
        gc[far] = cv2.GC_BGD
        gc[fov == 0] = cv2.GC_BGD
        gc[init > 0] = cv2.GC_PR_FGD
        core = cv2.erode(init, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        gc[core > 0] = cv2.GC_FGD
        if not (gc == cv2.GC_BGD).any() or not ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).any():
            return init
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(rgb, gc, None, bgd, fgd, self.cfg.grabcut_iters, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return init
        out = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        out = _fill_holes(_largest_component(out))
        return out if out.sum() > 0 else init

    def colour_classes(self, lab: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        lab_std = np.stack([lab[..., 0] * 100.0 / 255.0, lab[..., 1] - 128.0, lab[..., 2] - 128.0], -1)
        protos = np.array(list(STOLZ_COLOURS_LAB.values()), np.float32)
        d = np.linalg.norm(lab_std[..., None, :] - protos[None, None], axis=-1)
        cls = d.argmin(-1)
        cls[(d.min(-1) > self.cfg.colour_max_delta_e) | (mask == 0)] = -1
        area = max(int(mask.sum()), 1)
        fracs = {name: float((cls == i).sum()) / area for i, name in enumerate(STOLZ_COLOURS_LAB)}
        return cls, fracs

    def _principal_frame(self, mask: np.ndarray) -> np.ndarray:
        m = cv2.moments(mask.astype(np.float32))
        cx, cy = m["m10"] / (m["m00"] + EPS), m["m01"] / (m["m00"] + EPS)
        theta = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
        h, w = mask.shape
        rot = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        rot[0, 2] += w / 2.0 - cx
        rot[1, 2] += h / 2.0 - cy
        return rot

    def asymmetry(self, mask: np.ndarray, cls: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = mask.shape
        rot = self._principal_frame(mask)
        inv = cv2.invertAffineTransform(rot)
        m_r = cv2.warpAffine(mask.astype(np.float32), rot, (w, h), flags=cv2.INTER_NEAREST)
        c_r = cv2.warpAffine((cls + 1).astype(np.float32), rot, (w, h), flags=cv2.INTER_NEAREST)
        area = m_r.sum() + EPS
        yy, xx = np.mgrid[0:h, 0:w]
        halves = ((np.flipud, yy < h / 2.0), (np.fliplr, xx < w / 2.0))
        score, xor_total = 0.0, np.zeros_like(m_r)
        for flip, half in halves:
            xor = np.abs(m_r - flip(m_r))
            shape_asym = xor.sum() / (2 * area)
            side_1 = (m_r > 0) & half
            side_2 = (m_r > 0) & ~half
            hist_1 = np.bincount(c_r[side_1].astype(int), minlength=7)[1:].astype(np.float32)
            hist_2 = np.bincount(c_r[side_2].astype(int), minlength=7)[1:].astype(np.float32)
            colour_asym = 0.5 * np.abs(hist_1 / (hist_1.sum() + EPS) - hist_2 / (hist_2.sum() + EPS)).sum()
            colour_xor = ((c_r != flip(c_r)) & (m_r > 0) & (flip(m_r) > 0)).astype(np.float32)
            if shape_asym > self.cfg.asym_shape_tau or colour_asym > self.cfg.asym_colour_tau:
                score += 1.0
            xor_total += xor + 0.5 * colour_xor
        back = cv2.warpAffine(xor_total, inv, (w, h), flags=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(back, (0, 0), 2.0), score

    def border(self, lab: np.ndarray, mask: np.ndarray, skin: np.ndarray) -> tuple[np.ndarray, float]:
        lum = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.0)
        grad = np.hypot(cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3))
        skin_std = max(float(lum[skin > 0].std()) if skin.any() else float(lum.std()), 1e-2)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.zeros_like(lum), 0.0
        pts = max(contours, key=cv2.contourArea)[:, 0, :]
        m = cv2.moments(mask.astype(np.float32))
        cx, cy = m["m10"] / (m["m00"] + EPS), m["m01"] / (m["m00"] + EPS)
        ang = (np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx) + np.pi) / (2 * np.pi)
        octant = np.minimum((ang * 8).astype(int), 7)
        sharp = grad[pts[:, 1], pts[:, 0]] / (skin_std + EPS)
        octant_sharp = np.array([np.median(sharp[octant == k]) if np.any(octant == k) else 0.0 for k in range(8)])
        score = float((octant_sharp > self.cfg.border_abrupt_tau).sum())
        band = cv2.dilate(mask, np.ones((7, 7), np.uint8)) - cv2.erode(mask, np.ones((7, 7), np.uint8))
        bmap = np.zeros_like(lum)
        ys, xs = np.nonzero(band)
        a_band = (np.arctan2(ys - cy, xs - cx) + np.pi) / (2 * np.pi)
        bmap[ys, xs] = octant_sharp[np.minimum((a_band * 8).astype(int), 7)] * grad[ys, xs] / (skin_std + EPS)
        return cv2.GaussianBlur(bmap, (0, 0), 2.0), score

    def colour(self, cls: np.ndarray, fracs: dict[str, float], mask: np.ndarray) -> tuple[np.ndarray, float]:
        present = [i for i, n in enumerate(STOLZ_COLOURS_LAB) if fracs[n] >= self.cfg.colour_min_frac]
        score = float(max(len(present), 1))
        onehot = np.stack([(cls == i).astype(np.float32) for i in range(len(STOLZ_COLOURS_LAB))], 0)
        local = np.stack([cv2.GaussianBlur(c, (0, 0), 4.0) for c in onehot], 0)
        p = local / (local.sum(0, keepdims=True) + EPS)
        entropy = -(p * np.log(p + EPS)).sum(0)
        return entropy * mask, score

    def differential_structures(self, lab: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, dict[str, float]]:
        lum = lab[..., 0].astype(np.float32)
        inside = mask > 0
        area = max(int(inside.sum()), 1)
        maps: dict[str, np.ndarray] = {}
        present: dict[str, float] = {}

        def blobs(sigmas: tuple[float, ...]) -> tuple[np.ndarray, int]:
            resp = np.max([s ** 2 * cv2.Laplacian(cv2.GaussianBlur(lum, (0, 0), s), cv2.CV_32F) for s in sigmas], 0)
            peaks = (resp == cv2.dilate(resp, np.ones((5, 5), np.uint8))) & (resp > self.cfg.blob_tau) & inside
            return cv2.GaussianBlur(peaks.astype(np.float32), (0, 0), max(sigmas) * 1.5), int(peaks.sum())

        maps["dots"], n_dots = blobs(self.cfg.dot_sigmas)
        maps["globules"], n_glob = blobs(self.cfg.globule_sigmas)
        present["dots"] = float(n_dots >= self.cfg.min_dots)
        present["globules"] = float(n_glob >= self.cfg.min_globules)

        ridge = np.zeros_like(lum)
        theta = np.zeros_like(lum)
        for s in self.cfg.ridge_sigmas:
            l1, l2, th = _hessian_eigs(lum, s)
            r = np.where((l2 > 0) & (np.abs(l1) < 0.5 * l2), l2, 0.0)
            upd = r > ridge
            ridge[upd], theta[upd] = r[upd], th[upd]
        ridge_bin = ((ridge > self.cfg.ridge_tau) & inside).astype(np.uint8)
        holes = 0
        contours, hier = cv2.findContours(ridge_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is not None:
            holes = int((hier[0][:, 3] >= 0).sum())
        net_frac = ridge_bin.sum() / area
        present["pigment_network"] = float(net_frac >= self.cfg.network_min_frac and holes >= self.cfg.network_min_holes)
        maps["pigment_network"] = ridge * inside

        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        periphery = inside & (dist < 0.3 * (dist.max() + EPS))
        m = cv2.moments(mask.astype(np.float32))
        cx, cy = m["m10"] / (m["m00"] + EPS), m["m01"] / (m["m00"] + EPS)
        yy, xx = np.mgrid[0:lum.shape[0], 0:lum.shape[1]]
        radial = np.arctan2(yy - cy, xx - cx)
        aligned = np.abs(np.cos(theta - radial)) > 0.8
        streak_px = (ridge_bin > 0) & aligned & periphery
        present["streaks"] = float(streak_px.sum() / area >= self.cfg.streak_min_frac)
        maps["streaks"] = cv2.GaussianBlur(streak_px.astype(np.float32) * ridge, (0, 0), 1.5)

        mean = cv2.blur(lum, (15, 15))
        local_std = np.sqrt(np.maximum(cv2.blur(lum * lum, (15, 15)) - mean * mean, 0.0))
        skin_l = float(lum[~inside].mean()) if (~inside).any() else float(lum.mean())
        flat = inside & (local_std < self.cfg.structureless_tau) & (lum < skin_l - 10.0)
        present["structureless"] = float(flat.sum() / area >= self.cfg.structureless_min_frac)
        maps["structureless"] = cv2.GaussianBlur(flat.astype(np.float32), (0, 0), 3.0)

        score = float(max(sum(present.values()), 1.0))
        d_map = sum(_norm01(maps[k]) * present[k] for k in maps) if sum(present.values()) > 0 else _norm01(ridge * inside)
        return np.asarray(d_map, np.float32) * mask, score, present

    def generate(self, rgb: np.ndarray, binary_mask: np.ndarray | None = None) -> CSMResult:
        if rgb.dtype != np.uint8 or rgb.ndim != 3:
            raise ValueError("rgb must be uint8 (H, W, 3)")
        fov = self.fov_mask(rgb)
        clean = self.remove_hair(rgb)
        lab = self.flatfield_lab(clean, fov)
        if binary_mask is None:
            mask, quality, channel = self.segment(clean, lab, fov)
        else:
            mask, quality, channel = (binary_mask > 0).astype(np.uint8), 1.0, "external"
        if mask.sum() < 10:
            zero = np.zeros(mask.shape, np.float32)
            return CSMResult(zero, {}, mask, fov, 0.0, {"TDS": 0.0}, channel)
        skin = ((fov > 0) & (cv2.dilate(mask, np.ones((15, 15), np.uint8)) == 0)).astype(np.uint8)

        cls, fracs = self.colour_classes(lab, mask)
        m_a, s_a = self.asymmetry(mask, cls)
        m_b, s_b = self.border(lab, mask, skin)
        m_c, s_c = self.colour(cls, fracs, mask)
        m_d, s_d, d_present = self.differential_structures(lab, mask)

        wa, wb, wc, wd = self.cfg.tds_weights
        contrib = np.array([wa * s_a, wb * s_b, wc * s_c, wd * s_d], np.float32)
        tds = float(contrib.sum())
        share = contrib / (tds + EPS)
        comps = {"A": _norm01(m_a), "B": _norm01(m_b), "C": _norm01(m_c), "D": _norm01(m_d)}
        saliency = sum(share[i] * comps[k] for i, k in enumerate("ABCD"))
        saliency = _norm01(np.asarray(saliency, np.float32)) * (fov > 0)

        scores = {"A": s_a, "B": s_b, "C": s_c, "D": s_d, "TDS": tds}
        scores.update({f"D_{k}": v for k, v in d_present.items()})
        scores.update({f"C_{k}": v for k, v in fracs.items()})
        return CSMResult(saliency.astype(np.float32), comps, mask, fov, float(quality), scores, channel)
