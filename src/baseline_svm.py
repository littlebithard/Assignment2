import numpy as np
from skimage.feature import hog
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .common import load_image, metrics


def features(path):
    img = load_image(path).resize((128, 128))
    h = hog(np.asarray(img.convert("L")), orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    hsv = np.asarray(img.convert("HSV"))
    hist = np.concatenate([np.histogram(hsv[..., i], bins=16, range=(0, 256))[0] for i in range(3)]).astype(float)
    return np.concatenate([h, hist / hist.sum()])


def cv_svm(df, col="path", folds=5, seed=42, C=0.1):
    X = np.stack([features(p) for p in df[col]]); y = df.y.values
    out = []
    for k, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y)):
        pipe = Pipeline([("s", StandardScaler()), ("svm", SVC(kernel="rbf", C=C, gamma="scale"))]).fit(X[tr], y[tr])
        out.append(dict(config=f"SVM baseline ({'masked' if col == 'mpath' else 'raw'})", fold=k, **metrics(y[te], pipe.predict(X[te]))))
    return out
