# Chocolate Bar Recognition II — SOTA upgrade (CV Assignment 2)

**Mirzhan Myssyrgaliyev · SE-2414 · Astana IT University · Computer Vision (Nurym Kaiyrkhan)**

Continuation of Assignment 1: four chocolate bars (Kazakhstan, KitKat, Milky Way, Snickers), 800 frames from
16 self-recorded phone videos at 4 locations. Assignment 2 adds SOTA components, robust evaluation,
explainability and a demo app.

| Deliverable | Where |
|---|---|
| Report (8 pages) | [`report/A2_Report_Mirzhan_Myssyrgaliyev.pdf`](report/A2_Report_Mirzhan_Myssyrgaliyev.pdf) |
| Training + evaluation notebook (Colab) | [`A2_chocolate_sota.ipynb`](A2_chocolate_sota.ipynb) |
| Demo app  | [`app/`](app/) — run guide: [`app/RUN_LOCALLY.md`](app/RUN_LOCALLY.md) |

## What was added in A2

- **DINOv2 with registers** (ViT-S/14) as a SOTA backbone, compared with the A1 MobileNetV2 baseline.
- **SAM 2.1** segmentation → bars pasted onto new DTD texture backgrounds → labelled **SYNTHETIC** training data.
- **SigLIP** zero-shot reference (no training).
- **Leave-one-location-out** evaluation (4 folds), **stress test** (blur, low light, partial view, occlusion, noise),
  Top-2, retrieval P@5, latency.
- **Explainability:** Grad-CAM (CNN and ViT) and a nearest-neighbour gallery.
- **Ablation:** with vs without synthetic data, for both backbones.

## Results

| Model | Accuracy, A1 split | Accuracy, LOLO mean ± std | Stress test (mean of 5 corruptions) |
|---|---|---|---|
| MobileNetV2 (A2 baseline) | 0.995 | 0.990 ± 0.014 | 0.935 |
| MobileNetV2 + SAM 2 synthetic | 0.985 | 0.989 ± 0.008 | 0.945 |
| DINOv2-reg | 0.995 | 0.988 ± 0.010 | 0.936 |
| DINOv2-reg + SAM 2 synthetic | 0.995 | 0.989 ± 0.006 | 0.920 |
| SigLIP zero-shot | 0.860 | 0.802 ± 0.078 | — |

All trained models are at the ceiling (~99%). DINOv2 gives cleaner Grad-CAM maps, better retrieval
(P@5 0.991 vs 0.981) and the best robustness to partial views and occlusion, at 6× the CPU latency.
Synthetic data roughly halves the spread across locations. Details in the report.

## Run the demo

```bash
cd app
uv venv --python 3.12 && source .venv/bin/activate   # Python 3.10+ required
uv pip install gradio==5.50.0 -r requirements.txt
python app.py            # http://127.0.0.1:7860
python app.py --share    # optional temporary public link
```

Upload a photo or use the webcam → predicted bar, Grad-CAM heatmap, 5 nearest training images, latency.

## Reproduce training

Open `A2_chocolate_sota.ipynb` in Google Colab, select a **T4 GPU**, set `DRIVE_PATH` to the dataset folder and
*Run all* (~40 min). Outputs: `figures.zip` (tables, figures, `results.json`) and `app_model.zip` (the `app/model`
and `app/examples` folders).

## Repository layout

```
├── A2_chocolate_sota.ipynb   training, evaluation, figures, model export
├── report/                   PDF and DOCX report
└── app/
    ├── app.py                Gradio demo
    ├── requirements.txt
    ├── RUN_LOCALLY.md        local run guide
    ├── model/                trained DINOv2 classifier, config, nearest-neighbour gallery
    └── examples/             held-out test images, labelled SYNTHETIC examples
```

Brand names belong to their owners; non-commercial student project.
