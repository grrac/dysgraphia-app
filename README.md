# Dysgraphia Risk Assessment

A preliminary screening tool that grades handwriting samples as **Low**, **Moderate**, or
**High** dysgraphia risk. Built as a Final Year Project, deployed with Streamlit.

> ⚠️ **Not a diagnostic tool.** This is a first-pass screener for research/demo purposes and
> does not constitute a clinical diagnosis. Always consult a certified specialist.

**Live app:** https://dysgraphia-detection-fyp2.streamlit.app

## How it works

1. **Upload** a handwriting sample.
2. **Classify** — the image is cleaned (ruling-line removal), normalized, and resized to a
   300×700 canvas, then passed to the model.
3. **Review** — the app returns a risk grade with a confidence score and supporting notes.

## Model

- **Pipeline:** image → ResNet-18 feature extractor (512-d, ImageNet weights, no fine-tuning)
  → `StandardScaler` → linear `SVC` (probability enabled).
- **Data:** handwriting from 60 respondents across three severity levels (Combined-60 setup,
  following the H2FCD methodology).
- **Reported performance (5-fold CV, image features + SVM):**

  | Metric | Score |
  |---|---|
  | Accuracy | 80.0% |
  | Macro-F1 | 75.5% |
  | High-Risk recall | 56.7% |

- **Known limitation:** High-Risk recall (~57%) means roughly half of severe cases are not
  flagged. This model was chosen precisely because it had the *highest* High-Risk recall of the
  configurations tested — for a screener, catching severe cases matters most — but it remains a
  screening aid, not a diagnostic. Results are only reliable on dataset-format input (white-on-black, a few combined lines);
  arbitrary photographs of handwriting are out of distribution and not a validated use case.

## Input format

The model was trained on **dataset-format** samples, and that is the input it expects:

- **White handwriting strokes on a solid black background** (not dark ink on white paper).
- **A few lines of writing** combined into one image — *not* a full page. This mirrors the
  3-line combined samples used in training.
- Roughly the framing produced by the project's preprocessing (ruling lines removed, content
  sized to a 300×700 canvas).

**Upload samples in this format for meaningful results.** The app also has an optional
"Phone photo (paper)" mode that converts an ordinary ink-on-paper photo to white-on-black, but
this is an approximation: it does not reproduce how the dataset was captured, and full-page
photos in particular fall well outside the training distribution. **Treat phone-photo
predictions as indicative only, not validated** — the reported accuracy figures apply to
dataset-format input.

## Project structure
dysgraphia-app/
├── app.py                 # Streamlit app (UI + inference)
├── style.css              # single injected stylesheet
├── requirements.txt
├── runtime.txt            # pins Python 3.12 for Streamlit Cloud
├── index.html             # standalone design mockup (not used at runtime)
├── .streamlit/
│   └── config.toml        # theme
├── src/
│   ├── init.py
│   └── preprocess.py      # ruling-line removal, resize/pad
└── model/
├── svm.pkl
├── scaler.pkl
└── deploy_config.json
## Run locally

Requires **Python 3.12**.

```bash
py -3.12 -m venv .venv          # Windows;  python3.12 -m venv .venv on macOS/Linux
.venv\Scripts\activate          # source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

The ResNet-18 weights (~45 MB) download automatically on first run.

## Deployment

Deployed on [Streamlit Community Cloud](https://share.streamlit.io). `runtime.txt` pins Python
3.12 so the PyTorch CPU wheels resolve; `requirements.txt` uses the PyTorch CPU index to keep
the build small.

## Acknowledgements

Methodology adapted from the H2FCD paper (Ramlan et al., *IET Image Processing*, 2026).

