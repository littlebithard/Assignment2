import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import scan_dataset, load_image
from src.segment import _require_cv2
import cv2

_require_cv2()
from src.segment import overlay_mask

PICK = ("01", "05", "09")  # three photos per class


def grabcut_mask(img, inset=0.06, iters=5):
    arr = np.asarray(img)[:, :, ::-1].copy()
    h, w = arr.shape[:2]
    rect = (int(inset * w), int(inset * h), int((1 - 2 * inset) * w), int((1 - 2 * inset) * h))
    gc = np.zeros((h, w), np.uint8)
    cv2.grabCut(arr, gc, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                iters, cv2.GC_INIT_WITH_RECT)
    m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = (lab == keep).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m.astype(bool)


def main():
    df, classes = scan_dataset(ROOT / "dataset")
    out_root = ROOT / "masks_gt"
    preview = []
    for c in classes:
        sub = df[df.label == c]
        chosen = []
        for tag in PICK:
            hit = [r for _, r in sub.iterrows() if tag in Path(r.path).stem]
            if hit:
                chosen.append(hit[0])
        if len(chosen) < 3:
            chosen = [r for _, r in sub.head(3).iterrows()]
        for r in chosen:
            img = load_image(r.path)
            mask = grabcut_mask(img)
            d = out_root / r.label
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(d / f"{Path(r.path).stem}.png")
            preview.append((c, img, mask))
            print(f"gt {r.label}/{Path(r.path).stem}  fg={mask.mean():.2f}")
    # preview grid: original | overlay
    import matplotlib.pyplot as plt
    n = len(preview)
    cols = 7
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    ax = np.atleast_2d(ax)
    for i, (c, img, mask) in enumerate(preview):
        r, col = divmod(i, cols)
        ax[r, col].imshow(overlay_mask(img.resize((256, 256)),
                                       np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((256, 256), Image.NEAREST)) > 127))
        ax[r, col].set_title(c, fontsize=8)
        ax[r, col].axis("off")
    for j in range(n, rows * cols):
        r, col = divmod(j, cols)
        ax[r, col].axis("off")
    fig.suptitle("GrabCut proxy GT overlays (independent of MedSAM2)")
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "gt_preview.png", dpi=120)
    print(f"wrote {n} GT masks -> {out_root}")


if __name__ == "__main__":
    main()
