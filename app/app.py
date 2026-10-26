"""Chocolate Bar Recognition — A2 demo (DINOv2 + Grad-CAM + nearest neighbours).

Upload a photo or use the camera -> predicted bar, Grad-CAM heatmap, 5 most similar training images, latency.
Model files come from the Colab notebook export (app_model.zip -> model/ and examples/).
"""
import os, json, time, math, platform, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import gradio as gr
from PIL import Image
from matplotlib import colormaps
from transformers import AutoConfig, AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
CFG = json.load(open(os.path.join(MODEL_DIR, "config.json")))
CLASSES, IMG_SIZE = CFG["classes"], CFG["img_size"]
MEAN, STD = torch.tensor(CFG["mean"]).view(3, 1, 1), torch.tensor(CFG["std"]).view(3, 1, 1)
PRETTY = {"kazakhstan": "Kazakhstan", "kitkat": "KitKat", "milkyway": "Milky Way", "snickers": "Snickers"}
pretty = lambda c: PRETTY.get(c, c)


class DinoClassifier(nn.Module):  # identical to the notebook
    def __init__(self, n_cls, backbone):
        super().__init__()
        self.backbone = backbone
        d = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d, n_cls))
        self.n_skip = 1 + getattr(self.backbone.config, "num_register_tokens", 0)  # CLS + register tokens

    def embed(self, x):
        h = self.backbone(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, self.n_skip:].mean(1)], dim=1)

    def forward(self, x):
        return self.head(self.embed(x))


backbone = AutoModel.from_config(AutoConfig.from_pretrained(os.path.join(MODEL_DIR, "dino_config")))
model = DinoClassifier(len(CLASSES), backbone)

# transformers v4 and v5 name the attention weights differently -> rename the checkpoint keys to whatever this version expects
_V5_TO_V4 = {".attention.q_proj.": ".attention.attention.query.", ".attention.k_proj.": ".attention.attention.key.",
             ".attention.v_proj.": ".attention.attention.value.", ".attention.o_proj.": ".attention.output.dense."}


def _rename(sd, mapping):
    out = {}
    for k, v in sd.items():
        for a, b in mapping.items():
            k = k.replace(a, b)
        out[k] = v
    return out


state = torch.load(os.path.join(MODEL_DIR, "dino_classifier.pt"), map_location="cpu")
expected = set(model.state_dict())
for mapping in ({}, _V5_TO_V4, {b: a for a, b in _V5_TO_V4.items()}):
    if set(_rename(state, mapping)) == expected:
        state = _rename(state, mapping)
        break
model.load_state_dict(state)
model.eval()
with torch.no_grad():  # warm-up so the first user request does not show the one-off start-up cost
    model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))

G = np.load(os.path.join(MODEL_DIR, "gallery.npz"))
G_EMB = torch.tensor(G["emb"].astype(np.float32))
G_EMB = G_EMB / G_EMB.norm(dim=1, keepdim=True)
G_Y, G_THUMBS = G["y"], G["thumbs"]
NN_THRESHOLD = CFG.get("nn_threshold", 0.0)
LAT = []  # forward-pass latencies of this session (ms)

DEVICE_INFO = (f"CPU: {platform.processor() or platform.machine()}, {torch.get_num_threads()} threads · "
               f"torch {torch.__version__} · Python {platform.python_version()}")


def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)  # centre-crop to a square, like the training frames
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((IMG_SIZE, IMG_SIZE))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return ((x - MEAN) / STD)[None], img


def grad_cam(x):
    store = {}

    def hook(_, __, out):
        out.retain_grad()
        store["a"] = out

    h = model.backbone.encoder.layer[-1].norm1.register_forward_hook(hook)
    x = x.clone().requires_grad_(True)
    logits = model(x)
    cls = int(logits.argmax(1))
    model.zero_grad()
    logits[0, cls].backward()
    h.remove()
    k = model.n_skip
    a, g = store["a"][0, k:].detach(), store["a"].grad[0, k:]
    s = int(math.sqrt(a.shape[0]))
    cam = F.relu(a @ g.mean(0)).reshape(1, 1, s, s)
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0].numpy()
    return cam / (cam.max() + 1e-8)


def overlay(img, cam, alpha=0.45):
    heat = colormaps["jet"](cam)[..., :3] * 255
    return Image.fromarray((np.asarray(img, dtype=np.float32) * (1 - alpha) + heat * alpha).astype(np.uint8))


def predict(img, request: gr.Request = None):
    if img is None:
        return None, None, [], "Upload a photo or take one with the camera."
    x, shown = preprocess(img)

    t0 = time.perf_counter()
    with torch.no_grad():
        emb = model.embed(x)
        probs = F.softmax(model.head(emb), 1)[0].numpy()
    ms = (time.perf_counter() - t0) * 1000
    LAT.append(ms)

    t1 = time.perf_counter()
    cam = grad_cam(x)
    cam_ms = (time.perf_counter() - t1) * 1000

    q = emb / emb.norm(dim=1, keepdim=True)
    sims = (q @ G_EMB.T)[0].numpy()
    nn_idx = np.argsort(-sims)[:5]
    gallery = [(G_THUMBS[i], f"{pretty(CLASSES[G_Y[i]])} {sims[i]:.2f}") for i in nn_idx]

    top = int(probs.argmax())
    label = {pretty(c): float(p) for c, p in zip(CLASSES, probs)}
    warn = ""
    if probs[top] < 0.6 or sims[nn_idx[0]] < NN_THRESHOLD:
        warn = ("\n\n⚠️ **Low confidence** — the photo looks unlike the training images. "
                "It may not be one of the 4 bars, or the bar is too small / covered / blurred.")
    ua = (request.headers.get("user-agent", "unknown") if request is not None else "unknown")[:120]
    info = (f"### {pretty(CLASSES[top])} — {probs[top]:.0%}{warn}\n\n"
            f"| Measure | Value |\n|---|---|\n"
            f"| Inference (forward pass) | **{ms:.0f} ms** |\n"
            f"| Average over {len(LAT)} request(s) | {np.mean(LAT):.0f} ms |\n"
            f"| Grad-CAM (extra backward pass) | {cam_ms:.0f} ms |\n"
            f"| Nearest training image similarity | {sims[nn_idx[0]]:.2f} |\n"
            f"| Server | {DEVICE_INFO} |\n"
            f"| Your browser | `{ua}` |")
    return label, overlay(shown, cam), gallery, info


m = CFG.get("test_location_metrics", {})
lolo = CFG.get("lolo") or {}
lolo_txt = "\n".join(f"| {n} | {v['acc_mean']:.3f} | {v['f1_mean']:.3f} |" for n, v in lolo.items())
ABOUT = f"""
## Model
**{CFG.get('model_name', 'DINOv2-S')}** — DINOv2 ViT-S/14 with registers (Meta) + linear head, trained on 4 chocolate bars
(Kazakhstan, KitKat, Milky Way, Snickers) from a self-recorded dataset (16 phone videos, 4 locations).

On a location never seen in training: accuracy **{m.get('accuracy', float('nan')):.3f}**, macro-F1 **{m.get('f1', float('nan')):.3f}**,
Top-2 **{m.get('top2', float('nan')):.3f}**, retrieval P@5 **{m.get('p_at_5', float('nan')):.3f}**.

{"| Leave-one-location-out | accuracy | macro-F1 |" + chr(10) + "|---|---|---|" + chr(10) + lolo_txt if lolo else ""}

## Explanations
* **Grad-CAM** — which image regions pushed the model towards the predicted class (red = strong).
* **Nearest neighbours** — the 5 training images closest to your photo in the DINOv2 embedding space.

## Synthetic data
Part of the training data is **synthetic**: the bar was segmented with {CFG.get('mask_method', 'SAM 2')}
and pasted onto new texture backgrounds ({CFG.get('backgrounds', 'DTD')}). Synthetic images are used only for training,
never for testing, and every synthetic image shown in this app is labelled **SYNTHETIC** (see the *Synthetic data* tab).

## Ethics & privacy
* Uploaded photos are processed in memory and **not stored**.
* Training videos were recorded by the author; the only person visible is the author's hand (consent given).
  Please do not upload photos of other people without their consent.
* Brand names belong to their owners; this is a non-commercial student project.
* Limitations: one bar per class, one phone, 4 locations — expect mistakes on other packaging, lighting or cameras.
  The model always answers with one of 4 classes, so other objects get a low-confidence warning, not a rejection.
"""

examples = sorted(glob.glob(os.path.join(HERE, "examples", "test", "*.jpg")))
synthetic = sorted(glob.glob(os.path.join(HERE, "examples", "synthetic", "*.jpg")))

with gr.Blocks(title="Chocolate Bar Recognition") as demo:
    gr.Markdown("# 🍫 Chocolate Bar Recognition\nUpload a photo or use the camera → prediction → visual explanation.")
    with gr.Tab("Demo"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(sources=["upload", "webcam"], type="pil", label="Photo")
                btn = gr.Button("Recognise", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=inp, label="Held-out test images (unseen location)")
            with gr.Column():
                out_label = gr.Label(num_top_classes=4, label="Prediction")
                out_info = gr.Markdown()
        with gr.Row():
            out_cam = gr.Image(label="Grad-CAM (DINOv2, last block)", type="pil")
            out_nn = gr.Gallery(label="5 nearest training images", columns=5, height=220)
        btn.click(predict, inputs=inp, outputs=[out_label, out_cam, out_nn, out_info])
        inp.upload(predict, inputs=inp, outputs=[out_label, out_cam, out_nn, out_info])
    with gr.Tab("Synthetic data"):
        gr.Markdown("⚠️ **All images below are SYNTHETIC** — SAM 2 cut-outs of real bars pasted on new backgrounds. "
                    "Used only to enlarge the training set.")
        gr.Gallery(value=synthetic, columns=4, height="auto", label="SYNTHETIC training examples")
    with gr.Tab("About & ethics"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    import sys
    # python app.py          -> http://127.0.0.1:7860 (only on this computer)
    # python app.py --share  -> also a temporary public https://....gradio.live link (works while this computer runs the app)
    demo.launch(share="--share" in sys.argv)
