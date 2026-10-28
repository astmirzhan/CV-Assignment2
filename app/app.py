"""Chocolate Bar Recognition — A2 demo (DINOv2 + Grad-CAM + nearest neighbours).

Upload a photo or use the camera -> predicted bar, Grad-CAM heatmap, 5 most similar training images, latency.
Model files come from the Colab notebook export (app_model.zip -> model/ and examples/).

Rewritten with Flask + vanilla HTML/CSS/JS (no Gradio).
"""
import os, json, time, math, platform, glob, io, base64
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from matplotlib import colormaps
from transformers import AutoConfig, AutoModel
from flask import Flask, request, jsonify, send_from_directory

# ── paths & config ──────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
CFG = json.load(open(os.path.join(MODEL_DIR, "config.json")))
CLASSES, IMG_SIZE = CFG["classes"], CFG["img_size"]
MEAN = torch.tensor(CFG["mean"]).view(3, 1, 1)
STD  = torch.tensor(CFG["std"]).view(3, 1, 1)
PRETTY = {"kazakhstan": "Kazakhstan", "kitkat": "KitKat",
          "milkyway": "Milky Way", "snickers": "Snickers"}
pretty = lambda c: PRETTY.get(c, c)

# ── model ───────────────────────────────────────────────────────────────────
class DinoClassifier(nn.Module):
    def __init__(self, n_cls, backbone):
        super().__init__()
        self.backbone = backbone
        d = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d, n_cls))
        self.n_skip = 1 + getattr(self.backbone.config, "num_register_tokens", 0)

    def embed(self, x):
        h = self.backbone(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, self.n_skip:].mean(1)], dim=1)

    def forward(self, x):
        return self.head(self.embed(x))


backbone = AutoModel.from_config(
    AutoConfig.from_pretrained(os.path.join(MODEL_DIR, "dino_config"))
)
model = DinoClassifier(len(CLASSES), backbone)

_V5_TO_V4 = {
    ".attention.q_proj.": ".attention.attention.query.",
    ".attention.k_proj.": ".attention.attention.key.",
    ".attention.v_proj.": ".attention.attention.value.",
    ".attention.o_proj.": ".attention.output.dense.",
}

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
with torch.no_grad():
    model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))

G = np.load(os.path.join(MODEL_DIR, "gallery.npz"))
G_EMB = torch.tensor(G["emb"].astype(np.float32))
G_EMB = G_EMB / G_EMB.norm(dim=1, keepdim=True)
G_Y, G_THUMBS = G["y"], G["thumbs"]
NN_THRESHOLD = CFG.get("nn_threshold", 0.0)
LAT: list[float] = []

DEVICE_INFO = (
    f"CPU: {platform.processor() or platform.machine()}, "
    f"{torch.get_num_threads()} threads · torch {torch.__version__} · "
    f"Python {platform.python_version()}"
)

# ── image helpers ───────────────────────────────────────────────────────────
def pil_to_b64(img: Image.Image, fmt="JPEG", quality=85) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def preprocess(img: Image.Image):
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s)).resize((IMG_SIZE, IMG_SIZE))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return ((x - MEAN) / STD)[None], img


def grad_cam(x):
    store = {}
    def hook(_, __, out):
        out.retain_grad(); store["a"] = out
    h = model.backbone.encoder.layer[-1].norm1.register_forward_hook(hook)
    x = x.clone().requires_grad_(True)
    logits = model(x)
    cls = int(logits.argmax(1))
    model.zero_grad(); logits[0, cls].backward(); h.remove()
    k = model.n_skip
    a, g = store["a"][0, k:].detach(), store["a"].grad[0, k:]
    s = int(math.sqrt(a.shape[0]))
    cam = F.relu(a @ g.mean(0)).reshape(1, 1, s, s)
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                        align_corners=False)[0, 0].numpy()
    return cam / (cam.max() + 1e-8)


def overlay(img, cam, alpha=0.45):
    heat = colormaps["jet"](cam)[..., :3] * 255
    blended = np.asarray(img, dtype=np.float32) * (1 - alpha) + heat * alpha
    return Image.fromarray(blended.astype(np.uint8))


# ── about text (pre-built) ─────────────────────────────────────────────────
m = CFG.get("test_location_metrics", {})
lolo = CFG.get("lolo") or {}

# ── example images list ─────────────────────────────────────────────────────
examples  = sorted(glob.glob(os.path.join(HERE, "examples", "test", "*.jpg")))
synthetic = sorted(glob.glob(os.path.join(HERE, "examples", "synthetic", "*.jpg")))

# ── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)

@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(os.path.join(HERE, "static"), path)

@app.route("/examples/<path:path>")
def example_files(path):
    return send_from_directory(os.path.join(HERE, "examples"), path)

@app.route("/api/examples")
def api_examples():
    test_imgs = ["/examples/" + os.path.relpath(p, os.path.join(HERE, "examples"))
                 for p in examples]
    synth_imgs = ["/examples/" + os.path.relpath(p, os.path.join(HERE, "examples"))
                  for p in synthetic]
    return jsonify(test=test_imgs, synthetic=synth_imgs)

@app.route("/api/about")
def api_about():
    lolo_rows = [{"name": n, "acc": v["acc_mean"], "f1": v["f1_mean"]}
                 for n, v in lolo.items()]
    return jsonify(
        model_name=CFG.get("model_name", "DINOv2-S"),
        accuracy=m.get("accuracy"),
        f1=m.get("f1"),
        top2=m.get("top2"),
        p_at_5=m.get("p_at_5"),
        lolo=lolo_rows,
        mask_method=CFG.get("mask_method", "SAM 2"),
        backgrounds=CFG.get("backgrounds", "DTD"),
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    file = request.files.get("image")
    if not file:
        return jsonify(error="No image uploaded"), 400
    img = Image.open(file.stream)
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

    gallery = []
    for i in nn_idx:
        thumb = Image.fromarray(G_THUMBS[i])
        gallery.append({
            "image": pil_to_b64(thumb),
            "label": pretty(CLASSES[G_Y[i]]),
            "sim": round(float(sims[i]), 2),
        })

    top = int(probs.argmax())
    predictions = {pretty(c): round(float(p), 4) for c, p in zip(CLASSES, probs)}
    low_conf = float(probs[top]) < 0.6 or float(sims[nn_idx[0]]) < NN_THRESHOLD
    ua = (request.headers.get("User-Agent", "unknown"))[:120]

    return jsonify(
        predicted_class=pretty(CLASSES[top]),
        confidence=round(float(probs[top]), 4),
        predictions=predictions,
        low_confidence=low_conf,
        cam_image=pil_to_b64(overlay(shown, cam)),
        gallery=gallery,
        stats={
            "inference_ms": round(ms, 1),
            "avg_ms": round(float(np.mean(LAT)), 1),
            "num_requests": len(LAT),
            "gradcam_ms": round(cam_ms, 1),
            "nn_sim": round(float(sims[nn_idx[0]]), 2),
            "device": DEVICE_INFO,
            "user_agent": ua,
        },
    )


if __name__ == "__main__":
    import sys
    port = int(os.environ.get("PORT", 7860))
    debug = "--debug" in sys.argv
    app.run(host="0.0.0.0", port=port, debug=debug)
