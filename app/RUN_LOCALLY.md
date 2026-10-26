# How to run the demo (local run guide)

Requirements: Python 3.10–3.13, ~1.5 GB free disk (PyTorch). Works on macOS (incl. Apple Silicon), Linux and Windows. No GPU needed.

```bash
cd app                        # folder with app.py, model/, examples/
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install gradio==5.50.0 -r requirements.txt
python app.py                 # opens http://127.0.0.1:7860
```

Optional public link (temporary, works only while this computer keeps the app running):

```bash
python app.py --share         # prints a https://xxxx.gradio.live link
```

Usage: upload a photo or take one with the webcam → predicted bar + confidence, Grad-CAM heatmap,
5 nearest training images, inference latency (ms) and the server/browser environment.
Tabs *Synthetic data* and *About & ethics* show the labelled synthetic examples, metrics and ethics notes.

Files:
- `app.py` — Gradio app (DINOv2-with-registers classifier, Grad-CAM, nearest neighbours)
- `model/dino_classifier.pt` — trained weights (fp16), `model/dino_config/` — backbone config
- `model/gallery.npz` — embeddings + thumbnails of the training images for the nearest-neighbour gallery
- `model/config.json` — class names, preprocessing, metrics
- `examples/` — held-out test images and labelled SYNTHETIC examples
