import os, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.common import *
from src.segment import Segmenter, apply_mask, overlay_mask, iou_dice
from src.experiments import run_cv, summarise
from src.baseline_svm import cv_svm

SEED = 42
BACKEND = os.environ.get("SEG_BACKEND", "medsam2")
EPOCHS = (8, 8)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("weights").mkdir(exist_ok=True)


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150)
    plt.close()


def main():
    device = get_device()
    print(env_info())

    df, classes = scan_dataset("dataset")
    print(len(df), "images,", len(classes), "classes")
    print(df.label.value_counts().sort_index().to_string())

    seg = Segmenter(BACKEND)
    print("Segmenter:", seg.name)
    Path("masks").mkdir(exist_ok=True)
    seg_times, mpaths, fgfrac = [], [], []
    print("segmenting (warm-up)...")
    seg.predict(load_image(df.path[0]))
    for i, r in df.iterrows():
        img = load_image(r.path)
        mask, dt = seg.predict(img)
        seg_times.append(dt); fgfrac.append(mask.mean())
        out = Path("dataset_masked") / r.label; out.mkdir(parents=True, exist_ok=True)
        mp = out / (Path(r.path).stem + ".png")
        apply_mask(img, mask).save(mp); mpaths.append(str(mp))
        Image.fromarray(mask.astype(np.uint8) * 255).save(Path("masks") / f"{r.label}__{Path(r.path).stem}.png")
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(df)}  last {dt*1000:.0f} ms  fg={mask.mean():.2f}")
    df["mpath"] = mpaths
    print(f"segmentation: {np.mean(seg_times)*1000:.0f} ms/image   fg {np.mean(fgfrac):.2f} +/- {np.std(fgfrac):.2f}")

    fig, ax = plt.subplots(3, len(classes), figsize=(3 * len(classes), 9))
    for j, c in enumerate(classes):
        r = df[df.label == c].iloc[0]
        img = load_image(r.path)
        m = np.asarray(Image.open(Path("masks") / f"{c}__{Path(r.path).stem}.png")) > 127
        for i, (im, t) in enumerate([(img, "photo"), (overlay_mask(img, m), "MedSAM2 mask"), (Image.open(r.mpath), "classifier input")]):
            ax[i, j].imshow(im); ax[i, j].axis("off")
            ax[i, j].set_title(f"{c}\n{t}" if i == 0 else t, fontsize=9)
    savefig("mask_examples.png")

    gt_files = sorted(Path("masks_gt").rglob("*.png")) if Path("masks_gt").exists() else []
    rows = []
    for g in gt_files:
        cls = g.parent.name
        pm = Path("masks") / f"{cls}__{g.stem}.png"
        if not pm.exists():
            continue
        gt = np.asarray(Image.open(g).convert("L").resize(Image.open(pm).size, Image.NEAREST)) > 127
        iou, dice = iou_dice(np.asarray(Image.open(pm)) > 127, gt)
        rows.append(dict(cls=cls, file=g.stem, iou=iou, dice=dice))
    if rows:
        iou_df = pd.DataFrame(rows)
        iou_df.to_csv("results/mask_iou.csv", index=False)
        print(f"{len(iou_df)} GT images | mean IoU {iou_df.iou.mean():.3f} | mean Dice {iou_df.dice.mean():.3f}")
        print(iou_df.groupby("cls")[["iou", "dice"]].mean().round(3))
    else:
        print("No GT masks in masks_gt/")

    sample = load_image(df.path[0])
    fig, ax = plt.subplots(2, 6, figsize=(18, 6))
    ax[0, 0].imshow(sample.resize((IMG_SIZE, IMG_SIZE))); ax[0, 0].set_title("original")
    for a in ax.ravel()[1:]:
        a.imshow(VIS_AUG(sample)); a.set_title("augmented")
    for a in ax.ravel():
        a.axis("off")
    savefig("augmentation_examples.png")
    n_real = len(df); n_train_fold = int(n_real * 0.8)
    print(f"real photos: {n_real} | train/fold ~{n_train_fold} | synthetic: 0")

    t0 = time.time()
    res, ex = run_cv(df, len(classes), folds=5, seed=SEED, epochs=EPOCHS, device=device)
    svm_rows = pd.DataFrame(cv_svm(df, "path", seed=SEED))
    res_all = pd.concat([svm_rows, res], ignore_index=True)
    res_all.to_csv("results/cv_per_fold.csv", index=False)
    print(f"CV done in {(time.time()-t0)/60:.1f} min")
    summary = summarise(res_all)
    summary.to_csv("results/cv_summary.csv")
    print(summary)

    g = res_all.groupby("config", sort=False)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(g)); w = 0.38
    ax.bar(x - w/2, g.accuracy.mean(), w, yerr=g.accuracy.std(), capsize=3, label="accuracy")
    ax.bar(x + w/2, g.f1.mean(), w, yerr=g.f1.std(), capsize=3, label="macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(list(g.mean(numeric_only=True).index), rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title("5-fold cross-validation (mean +/- std over folds)")
    savefig("cv_results.png")

    probe = res_all.dropna(subset=["acc_bg_removed"]).groupby("config")[["accuracy", "acc_bg_removed"]].mean().round(3)
    probe.columns = ["acc on raw test photos", "acc with background removed"]
    probe.to_csv("results/background_probe.csv")
    print(probe)

    fig, ax = plt.subplots(2, 4, figsize=(18, 7), sharey="row")
    for j, (name, h) in enumerate(ex["curves"].items()):
        ax[0, j].plot(h.epoch, h.train_loss, label="train")
        ax[0, j].plot(h.epoch, h.eval_loss, label="held-out fold")
        ax[1, j].plot(h.epoch, h.train_acc)
        ax[1, j].plot(h.epoch, h.eval_acc)
        ax[0, j].set_title(name); ax[1, j].set_xlabel("epoch")
        for a in ax[:, j]:
            a.axvline(8.5, color="gray", ls="--")
    ax[0, 0].set_ylabel("loss"); ax[1, 0].set_ylabel("accuracy"); ax[0, 0].legend()
    savefig("training_curves.png")

    def pooled(name):
        return (np.concatenate([a for a, _ in ex["preds"][name]]),
                np.concatenate([b for _, b in ex["preds"][name]]))

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for a, name in zip(ax, ["raw + aug (A1)", "masked + aug (A2)"]):
        y, p = pooled(name)
        cm = confusion_matrix(y, p, labels=range(len(classes)))
        a.imshow(cm, cmap="Blues"); a.set_title(name)
        a.set_xlabel("predicted"); a.set_ylabel("true")
        a.set_xticks(range(len(classes))); a.set_yticks(range(len(classes)))
        a.set_xticklabels(classes, rotation=45, ha="right"); a.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                a.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    savefig("confusion_matrices.png")
    y, p = pooled("masked + aug (A2)")
    print(classification_report(y, p, target_names=classes, zero_division=0))

    m0 = ex["fold0_models"]["masked + aug (A2)"]; te0 = ex["fold0_test"]
    cam_engine = GradCAM(m0)
    n = len(te0); cols = 7; rows_ = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows_ * 2, cols, figsize=(3 * cols, 6 * rows_))
    for k in range(rows_ * cols):
        r_, c_ = (k // cols) * 2, k % cols
        for a in (ax[r_, c_], ax[r_ + 1, c_]):
            a.axis("off")
        if k >= n:
            continue
        img = Image.open(te0.mpath[k]); x = EVAL_TF(img).unsqueeze(0).to(device)
        cam, idx, prob = cam_engine(x)
        ax[r_, c_].imshow(img.resize((IMG_SIZE, IMG_SIZE)))
        ax[r_ + 1, c_].imshow(cam_overlay(img, cam))
        ok = idx == te0.y[k]
        ax[r_, c_].set_title(f"true: {classes[te0.y[k]]}\npred: {classes[idx]}",
                             fontsize=8, color="green" if ok else "red")
    savefig("gradcam_examples.png")

    mdl = ex["fold0_models"]["masked + aug (A2)"]; mdl.eval(); cam_engine = GradCAM(mdl)
    seg_t, cls_t = [], []
    for pth in df.path.iloc[:20]:
        img = load_image(pth)
        mask, ts = seg.predict(img); seg_t.append(ts)
        t = time.perf_counter()
        x = EVAL_TF(apply_mask(img, mask)).unsqueeze(0).to(device); cam_engine(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        cls_t.append(time.perf_counter() - t)
    lat = pd.DataFrame({
        "stage": ["segmentation (" + seg.name + ")", "classification + Grad-CAM", "total"],
        "ms/image": [np.mean(seg_t) * 1000, np.mean(cls_t) * 1000, (np.mean(seg_t) + np.mean(cls_t)) * 1000],
    }).round(1)
    lat.to_csv("results/latency.csv", index=False)
    print(env_info()); print(lat)

    print("training final models on all 70 photos...")
    final_masked, _ = train_model(df, len(classes), augment=True, path_col="mpath",
                                  epochs_head=EPOCHS[0], epochs_ft=EPOCHS[1], seed=SEED)
    final_raw, _ = train_model(df, len(classes), augment=True, path_col="path",
                               epochs_head=EPOCHS[0], epochs_ft=EPOCHS[1], seed=SEED)
    torch.save(final_masked.state_dict(), "weights/cls_masked.pt")
    torch.save(final_raw.state_dict(), "weights/cls_raw.pt")
    json.dump(classes, open("weights/classes.json", "w"))
    print("saved weights/cls_masked.pt, cls_raw.pt, classes.json")
    print("DONE")


if __name__ == "__main__":
    main()
