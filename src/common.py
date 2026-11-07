import os, time, random, platform
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_image(path):
    """Open an image, apply the EXIF rotation stored by phone cameras, return RGB."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def scan_dataset(root):
    root = Path(root)
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(d.glob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append((str(f), d.name))
    df = pd.DataFrame(rows, columns=["path", "label"])
    classes = sorted(df.label.unique())
    df["y"] = df.label.map({c: i for i, c in enumerate(classes)})
    return df, classes


def env_info():
    dev = get_device()
    name = torch.cuda.get_device_name(0) if dev.type == "cuda" else dev.type.upper()
    return f"{platform.system()} {platform.release()} | Python {platform.python_version()} | torch {torch.__version__} | device: {name}"


AUG_OPS = [
    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(30),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
]
_NORM = [T.ToTensor(), T.Normalize(MEAN, STD)]


def make_transform(augment):
    if augment:
        return T.Compose(AUG_OPS + _NORM)
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE))] + _NORM)


EVAL_TF = make_transform(False)
VIS_AUG = T.Compose(AUG_OPS)  # PIL -> PIL, for figures and offline expansion


class PlantDS(Dataset):
    def __init__(self, frame, tf, path_col="path"):
        self.paths, self.y, self.tf = frame[path_col].tolist(), frame.y.tolist(), tf

    def __len__(self): return len(self.paths)

    def __getitem__(self, i): return self.tf(load_image(self.paths[i])), self.y[i]


def build_model(n_classes, pretrained=True, device=None):
    w = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v2(weights=w)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, n_classes))
    return m.to(device or get_device())


def _bn_eval(m):
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d): mod.eval()


def _epoch(model, loader, device, opt=None):
    train = opt is not None
    model.train(train)
    if train: _bn_eval(model)           # batches of 8 are too small for BN statistics
    tot, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return tot / n, correct / n


def train_model(train_df, n_classes, augment=True, path_col="path", eval_df=None, eval_col=None,
                epochs_head=8, epochs_ft=8, batch=8, seed=0, pretrained=True, device=None, verbose=False):
    """Two-phase transfer learning (head only, then last blocks), FIXED number of epochs.
    eval_df is only used to record curves; it never influences training."""
    device = device or get_device()
    seed_everything(seed)
    model = build_model(n_classes, pretrained, device)
    tl = DataLoader(PlantDS(train_df, make_transform(augment), path_col), batch_size=batch, shuffle=True)
    el = None
    if eval_df is not None:
        el = DataLoader(PlantDS(eval_df, EVAL_TF, eval_col or path_col), batch_size=batch)
    history = []

    def phase(params, lr, epochs, tag):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            a, b = _epoch(model, tl, device, opt)
            rec = dict(phase=tag, epoch=len(history) + 1, train_loss=a, train_acc=b)
            if el is not None:
                rec["eval_loss"], rec["eval_acc"] = _epoch(model, el, device)
            history.append(rec)
            if verbose: print(rec)

    for p in model.features.parameters(): p.requires_grad = False
    phase(model.classifier.parameters(), 1e-3, epochs_head, "head")
    for p in model.features[-5:].parameters(): p.requires_grad = True
    phase([p for p in model.parameters() if p.requires_grad], 1e-4, epochs_ft, "finetune")
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_df(model, frame, path_col="path", device=None, batch=8):
    device = device or get_device()
    model.eval()
    loader = DataLoader(PlantDS(frame, EVAL_TF, path_col), batch_size=batch)
    ys, ps, pr = [], [], []
    for x, y in loader:
        prob = torch.softmax(model(x.to(device)), 1).cpu()
        pr.append(prob); ps.append(prob.argmax(1)); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(pr).numpy()


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y, p), precision=pr, recall=rc, f1=f1)


class GradCAM:
    """Grad-CAM on the last conv block of MobileNetV2 (model.features[-1])."""
    def __init__(self, model, layer=None):
        self.model = model
        self.acts = self.grads = None
        (layer or model.features[-1]).register_forward_hook(self._fwd)

    def _fwd(self, mod, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            out = self.model(x)
            idx = int(out.argmax(1)) if class_idx is None else class_idx
            self.model.zero_grad()
            out[0, idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1, keepdim=True))
        cam = nn.functional.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), idx, torch.softmax(out.detach(), 1)[0].cpu().numpy()


def denorm(t):
    return (t.cpu() * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]).clamp(0, 1).permute(1, 2, 0).numpy()


def cam_overlay(pil_img, cam, alpha=0.45):
    """Return an RGB uint8 array: 224x224 resized image with the heat-map on top."""
    import matplotlib
    base = np.asarray(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(float) / 255
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return (np.clip((1 - alpha) * base + alpha * heat, 0, 1) * 255).astype(np.uint8)
