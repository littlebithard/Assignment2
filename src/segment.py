import os, time
from importlib import import_module
from pathlib import Path
import numpy as np
from PIL import Image

try:
    cv2 = import_module("cv2")
except ImportError:
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise ImportError("OpenCV is required for segmentation; install it with 'pip install opencv-python'.")


BG_COLOUR = (124, 116, 104)  # ImageNet mean colour: neutral background for masked images

DEFAULTS = {
    "medsam2": (os.environ.get("MEDSAM2_CFG", "configs/sam2.1_hiera_t512.yaml"),
                os.environ.get("MEDSAM2_CKPT", "weights/MedSAM2_latest.pt")),
    "sam2":    (os.environ.get("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_t.yaml"),
                os.environ.get("SAM2_CKPT", "weights/sam2.1_hiera_tiny.pt")),
}


def clean_mask(m, min_speck=0.002, max_hole=0.02):
    """Remove tiny specks and fill small holes. Large disconnected parts (extra leaves) are kept."""
    m = m.astype(np.uint8)
    area = m.size
    for val, limit in ((1, min_speck), (0, max_hole)):      # specks of 1s, then holes of 0s
        n, lab, stats, _ = cv2.connectedComponentsWithStats((m == val).astype(np.uint8), connectivity=8)
        for k in range(1, n):
            small = stats[k, cv2.CC_STAT_AREA] < limit * area
            if val == 0:  # only fill holes that do not touch the image border
                x, y, w, h = (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                              stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])
                small = small and x > 0 and y > 0 and x + w < m.shape[1] and y + h < m.shape[0]
            if small: m[lab == k] = 1 - val
    return m.astype(bool)


def apply_mask(img, mask, crop=True, pad=0.05, bg=BG_COLOUR):
    """Replace the background by a neutral colour and (optionally) crop to the plant's bounding box."""
    arr = np.asarray(img).copy()
    arr[~mask] = bg
    out = Image.fromarray(arr)
    if crop and mask.any():
        ys, xs = np.where(mask)
        h, w = mask.shape
        py, px = int(pad * h), int(pad * w)
        out = out.crop((max(xs.min() - px, 0), max(ys.min() - py, 0), min(xs.max() + px, w), min(ys.max() + py, h)))
    return out


def overlay_mask(img, mask, colour=(30, 200, 90), alpha=0.5):
    arr = np.asarray(img).astype(float)
    arr[mask] = (1 - alpha) * arr[mask] + alpha * np.array(colour)
    return Image.fromarray(arr.astype(np.uint8))


def iou_dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) else 1.0
    return float(iou), float(dice)


class Segmenter:
    """backend: 'medsam2' | 'sam2' | 'grabcut'.  predict(img) -> (bool mask HxW, seconds)."""
    def __init__(self, backend="medsam2", checkpoint=None, config=None, device=None, inset=0.04):
        self.backend, self.inset = backend, inset
        self.name = {"medsam2": "MedSAM2", "sam2": "SAM 2.1 (tiny)", "grabcut": "GrabCut (fallback, not SOTA)"}[backend]
        if backend in ("medsam2", "sam2"):
            import torch
            build_sam2 = import_module("sam2.build_sam").build_sam2
            SAM2ImagePredictor = import_module("sam2.sam2_image_predictor").SAM2ImagePredictor
            cfg, ckpt = DEFAULTS[backend]
            cfg, ckpt = config or cfg, checkpoint or ckpt
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}. See README.md > 'Download weights'.")
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.predictor = SAM2ImagePredictor(build_sam2(cfg, ckpt, device=self.device))
        elif backend != "grabcut":
            raise ValueError(backend)

    def _prompt(self, w, h):
        box = np.array([self.inset * w, self.inset * h, (1 - self.inset) * w, (1 - self.inset) * h], dtype=np.float32)
        point = np.array([[w / 2, h / 2]], dtype=np.float32)
        return box, point

    def predict(self, img):
        t0 = time.perf_counter()
        w, h = img.size
        if self.backend == "grabcut":
            mask = self._grabcut(img)
        else:
            import torch
            arr = np.asarray(img)
            box, point = self._prompt(w, h)
            use_amp = self.device == "cuda"
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                self.predictor.set_image(arr)
                masks, scores, _ = self.predictor.predict(point_coords=point, point_labels=np.array([1]),
                                                          box=box[None], multimask_output=True)
            mask = masks[int(np.argmax(scores))].astype(bool)
        mask = clean_mask(mask)
        if mask.mean() < 0.05:  # degenerate result: fall back to the prompt box itself
            box, _ = self._prompt(w, h)
            mask = np.zeros((h, w), bool)
            mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = True
        return mask, time.perf_counter() - t0

    def _grabcut(self, img):
        scale = 512 / max(img.size)
        small = np.asarray(img.resize((int(img.width * scale), int(img.height * scale))))[:, :, ::-1].copy()
        h, w = small.shape[:2]
        rect = (int(self.inset * w), int(self.inset * h), int((1 - 2 * self.inset) * w), int((1 - 2 * self.inset) * h))
        gc = np.zeros(small.shape[:2], np.uint8)
        cv2.grabCut(small, gc, rect, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_RECT)
        m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        return cv2.resize(m, img.size, interpolation=cv2.INTER_NEAREST).astype(bool)
