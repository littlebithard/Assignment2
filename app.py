import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import gradio as gr
from PIL import Image

from src.common import (build_model, get_device, EVAL_TF, GradCAM, cam_overlay, env_info, IMG_SIZE)
from src.segment import Segmenter, apply_mask, overlay_mask

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="medsam2", choices=["medsam2", "sam2", "grabcut"])
ap.add_argument("--weights", help="Classifier weights directory (default: weights beside app.py)")
ap.add_argument("--share", action="store_true")
args = ap.parse_args()

W = Path(args.weights) if args.weights else Path(__file__).resolve().parent / "weights"
device = get_device()
classes_path = W / "classes.json"
if not classes_path.is_file():
    raise FileNotFoundError(
        f"Classifier metadata not found: {classes_path}. "
        "Train the classifier first by running the notebook's training cells "
        "or `python scripts/run_pipeline.py` from the project directory."
    )
classes = json.loads(classes_path.read_text())

def load_cls(name):
    f = W / name
    if not f.exists(): return None
    m = build_model(len(classes), pretrained=False, device=device)
    m.load_state_dict(torch.load(f, map_location=device)); m.eval()
    return m

MODELS = {"masked": load_cls("cls_masked.pt"), "raw": load_cls("cls_raw.pt")}
assert MODELS["masked"] is not None, "weights/cls_masked.pt not found. Run the notebook first."
seg = Segmenter(args.backend)
cams = {k: GradCAM(m) for k, m in MODELS.items() if m is not None}
LAT = {"seg": [], "cls": [], "total": []}
ENV = env_info() + f" | segmenter: {seg.name}"

NOTICE = ("**Ethics & disclosure.** The dataset contains only plants and flowers (no people, no faces). "
          "No generative AI or synthetic images are used: augmentation only applies random crops, flips, rotations "
          "and colour changes to real photos. Masks and heat-maps shown here are machine-generated explanations, "
          "not ground truth, and the model only knows the 7 classes it was trained on.")

def run(image, use_seg, request: gr.Request = None):
    if image is None: return None, None, None, None, "Upload or capture an image first."
    img = image.convert("RGB")
    t0 = time.perf_counter()
    if use_seg:
        mask, t_seg = seg.predict(img)
        model_key, model_in = "masked", apply_mask(img, mask)
        mask_vis = overlay_mask(img, mask)
    else:
        t_seg, model_key, model_in, mask_vis = 0.0, "raw" if MODELS["raw"] else "masked", img, img
    t1 = time.perf_counter()
    x = EVAL_TF(model_in).unsqueeze(0).to(device)
    cam, idx, prob = cams[model_key](x)
    if device.type == "cuda": torch.cuda.synchronize()
    t_cls = time.perf_counter() - t1
    total = time.perf_counter() - t0
    for k, v in (("seg", t_seg), ("cls", t_cls), ("total", total)): LAT[k].append(v)
    order = np.argsort(-prob)[:5]
    label = {classes[i]: float(prob[i]) for i in order}
    n = len(LAT["total"])
    ua = (request.headers.get("user-agent", "n/a") if request is not None else "n/a")
    stats = (f"**Latency (this image):** segmentation {t_seg*1000:.0f} ms | classification + Grad-CAM {t_cls*1000:.0f} ms | total {total*1000:.0f} ms  \n"
             f"**Average over {n} request(s):** segmentation {np.mean(LAT['seg'])*1000:.0f} ms | classification {np.mean(LAT['cls'])*1000:.0f} ms | total {np.mean(LAT['total'])*1000:.0f} ms  \n"
             f"**Server environment:** {ENV}  \n**Browser:** {ua}  \n"
             f"**Pipeline used:** {seg.name + ' mask -> ' if use_seg else ''}MobileNetV2 ({model_key} model)")
    return mask_vis, model_in, Image.fromarray(cam_overlay(model_in, cam)), label, stats

with gr.Blocks(title="Houseplant classifier: MedSAM2 + Grad-CAM") as demo:
    gr.Markdown("# Houseplant classifier with MedSAM2 segmentation and Grad-CAM\n"
                "Upload a plant photo (or use the webcam). The plant is segmented, classified, and the heat-map shows what the model used.")
    with gr.Row():
        inp = gr.Image(type="pil", sources=["upload", "webcam"], label="Photo")
        with gr.Column():
            use_seg = gr.Checkbox(value=True, label="Segment the plant first (MedSAM2)")
            btn = gr.Button("Classify", variant="primary")
    with gr.Row():
        o1 = gr.Image(label="1. Predicted mask", height=320)
        o2 = gr.Image(label="2. Classifier input", height=320)
        o3 = gr.Image(label="3. Grad-CAM heat-map", height=320)
    lab = gr.Label(label="Top-5 prediction", num_top_classes=5)
    info = gr.Markdown()
    gr.Markdown(NOTICE)
    btn.click(run, [inp, use_seg], [o1, o2, o3, lab, info])

if __name__ == "__main__":
    demo.launch(share=args.share)
