import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .common import train_model, predict_df, metrics

CONFIGS = {
    "raw, no aug":       dict(inp="path",  augment=False),
    "raw + aug (A1)":    dict(inp="path",  augment=True),
    "masked, no aug":    dict(inp="mpath", augment=False),
    "masked + aug (A2)": dict(inp="mpath", augment=True),
}


def run_cv(df, n_classes, configs=CONFIGS, folds=5, seed=42, epochs=(8, 8), pretrained=True, device=None, log=print):
    """df needs columns: path (raw), mpath (masked), y.
    Returns (per-fold DataFrame, extras) where extras has curves / pooled predictions / fold-0 models."""
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    rows, curves, preds, models0, test0 = [], {}, {}, {}, None
    for name, cfg in configs.items():
        for k, (tr, te) in enumerate(skf.split(df, df.y)):
            tr_df, te_df = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
            model, hist = train_model(tr_df, n_classes, augment=cfg["augment"], path_col=cfg["inp"],
                                      eval_df=te_df if k == 0 else None, eval_col=cfg["inp"],
                                      epochs_head=epochs[0], epochs_ft=epochs[1], seed=seed + k,
                                      pretrained=pretrained, device=device)
            y, p, _ = predict_df(model, te_df, cfg["inp"], device)
            rec = dict(config=name, fold=k, **metrics(y, p))
            if cfg["inp"] == "path":   # background-reliance probe: same model, plant on neutral background
                y2, p2, _ = predict_df(model, te_df, "mpath", device)
                rec["acc_bg_removed"] = metrics(y2, p2)["accuracy"]
            rows.append(rec)
            preds.setdefault(name, []).append((y, p))
            if k == 0:
                curves[name], models0[name], test0 = hist, model, te_df
            log(f"{name:20s} fold {k}: acc={rec['accuracy']:.3f} f1={rec['f1']:.3f}")
    return pd.DataFrame(rows), dict(curves=curves, preds=preds, fold0_models=models0, fold0_test=test0)


def summarise(res):
    g = res.groupby("config", sort=False)
    out = g[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"]).round(3)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    if "acc_bg_removed" in res:
        out["acc_bg_removed_mean"] = g["acc_bg_removed"].mean().round(3)
    return out
