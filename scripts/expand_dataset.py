import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.model_selection import train_test_split
from src.common import scan_dataset, load_image, VIS_AUG, seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset"); ap.add_argument("--out", default="dataset_expanded")
ap.add_argument("--k", type=int, default=10, help="augmented copies per training image")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

seed_everything(a.seed)
df, _ = scan_dataset(a.data)
train, test = train_test_split(df, test_size=0.2, stratify=df.y, random_state=a.seed)   # same 80/20 as notebook
out = Path(a.out)
for _, r in train.iterrows():
    img = load_image(r.path)
    d = out / "train" / r.label; d.mkdir(parents=True, exist_ok=True)
    img.resize((224, 224)).save(d / f"{Path(r.path).stem}_orig.jpg")
    for i in range(a.k):
        VIS_AUG(img).save(d / f"{Path(r.path).stem}_aug{i:02d}.jpg", quality=92)
for _, r in test.iterrows():
    d = out / "test" / r.label; d.mkdir(parents=True, exist_ok=True)
    load_image(r.path).resize((224, 224)).save(d / f"{Path(r.path).stem}.jpg")
n_tr = sum(1 for _ in (out / "train").rglob("*.jpg")); n_te = sum(1 for _ in (out / "test").rglob("*.jpg"))
print(f"real images: {len(df)} | expanded train set: {n_tr} (x{n_tr/len(train):.0f}) | untouched test set: {n_te}")
