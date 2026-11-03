# How to run the demo (local run guide)

Requirements: Python 3.10–3.13, ~1.5 GB free disk (PyTorch). Works on macOS (incl. Apple Silicon), Linux and Windows. No GPU needed.

```bash
cd app                        # folder with app.py, index.html, model/, examples/
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                 # open http://127.0.0.1:7860
```

The server listens on all interfaces, so a phone on the same Wi-Fi can open `http://<laptop-IP>:7860`
(the camera button needs HTTPS or localhost in most mobile browsers — use file upload there).
Port can be changed with `PORT=8000 python app.py`.

Usage: upload a photo, take one with the webcam or click an example → predicted bar + confidence, Grad-CAM heatmap,
5 nearest training images, inference latency (ms) and the server/browser environment.
Tabs *Synthetic data* and *About & ethics* show the labelled synthetic examples, metrics and ethics notes.

Files:
- `app.py` — Flask backend: DINOv2-with-registers classifier, Grad-CAM, nearest neighbours (`/api/predict`)
- `index.html` — web UI (plain HTML/CSS/JS, no build step)
- `model/dino_classifier.pt` — trained weights (fp16), `model/dino_config/` — backbone config
- `model/gallery.npz` — embeddings + thumbnails of the training images for the nearest-neighbour gallery
- `model/config.json` — class names, preprocessing, metrics
- `examples/` — held-out test images and labelled SYNTHETIC examples
