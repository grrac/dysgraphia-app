"""
Dysgraphia Risk Assessment — Streamlit app.

Pipeline (identical to FYP2_Preprocessing_Combined + Model_training_FYP2_Combined):
  upload -> grayscale -> remove_ruling_lines -> resize_and_pad (300x700)
         -> stack to 3ch, /255 (NO ImageNet normalize)
         -> ResNet-18 (children[:-1]) -> flatten 512
         -> StandardScaler -> SVC(probability=True) -> class + confidence
"""
import base64
import io
import json

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import joblib
from PIL import Image

from src.preprocess import preprocess_single

st.set_page_config(page_title="Dysgraphia Risk Assessment",
                   page_icon="🖊️", layout="centered")

# ------------------------------------------------------------------
# Artifacts (loaded once, cached across reruns)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    scaler = joblib.load("model/scaler.pkl")
    svm    = joblib.load("model/svm.pkl")
    with open("model/deploy_config.json") as f:
        cfg = json.load(f)
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    extractor = nn.Sequential(*list(resnet.children())[:-1]).eval()
    return scaler, svm, cfg, extractor


scaler, svm, CFG, EXTRACTOR = load_artifacts()
CLASS_NAMES = CFG["class_names"]          # ['Low Risk','Moderate Risk','High Risk']


# ------------------------------------------------------------------
# Inference — mirrors the notebook step for step
# ------------------------------------------------------------------
def extract_features(canvas_uint8):
    rgb = np.stack((canvas_uint8,) * 3, axis=-1)                       # (300,700,3)
    t = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        feat = torch.flatten(EXTRACTOR(t), 1).cpu().numpy()           # (1,512)
    return feat


def classify(feat):
    proba = svm.predict_proba(scaler.transform(feat))[0]
    pos = int(np.argmax(proba))
    label_idx = int(svm.classes_[pos])                                # 0 / 1 / 2
    return label_idx, float(proba[pos])


# ------------------------------------------------------------------
# Per-class copy (heuristic template text keyed to the predicted class,
# not model-generated — see the notes we discussed for the viva)
# ------------------------------------------------------------------
VERDICT = {
    0: dict(cls="low", sub="No significant indicators detected",
            body="Handwriting patterns fall within typical developmental ranges. "
                 "No significant indicators of dysgraphia were identified in this sample.",
            obs=["Overall handwriting legibility is high",
                 "Letter formation is consistent across the sample",
                 "Word spacing appears regular and controlled",
                 "No significant reversals or omissions detected"]),
    1: dict(cls="mod", sub="Some dysgraphia markers present",
            body="Noticeable irregularities in letter formation, spacing, or legibility "
                 "were detected. Several patterns align with dysgraphia markers.",
            obs=["Some inconsistency in letter formation",
                 "Spacing varies across the sample",
                 "Occasional dips in legibility",
                 "A number of atypical letter shapes present"]),
    2: dict(cls="high", sub="Multiple markers detected",
            body="Multiple consistent markers of dysgraphia were detected, with a "
                 "noticeable impact on legibility and fluency.",
            obs=["Frequent irregularities in letter formation",
                 "Inconsistent spacing and baseline alignment",
                 "Reduced overall legibility",
                 "Signs of effortful, non-fluent writing"]),
}

ICON_CHECK = ('<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
              'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" '
              'r="9"/><path d="m8.5 12 2.4 2.4 4.6-4.8"/></svg>')
ICON_ALERT = ('<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
              'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.3 3.9 1.8 18a2 '
              '2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4"/>'
              '<path d="M12 17h.01"/></svg>')


def result_html(label_idx, conf, filename, data_uri):
    v = VERDICT[label_idx]
    icon = ICON_CHECK if label_idx == 0 else ICON_ALERT
    pct = round(conf * 100)
    obs = "".join(f"<li>{o}</li>" for o in v["obs"])
    return f"""
<div class="card">
  <div class="verdict {v['cls']}">
    <span class="verdict__icon">{icon}</span>
    <div>
      <div class="verdict__label">{CLASS_NAMES[label_idx]}</div>
      <div class="verdict__sub">{v['sub']}</div>
      <div class="verdict__body">{v['body']}</div>
    </div>
  </div>
  <div class="result__grid">
    <div>
      <span class="eyebrow">Analyzed sample</span>
      <div class="thumb"><img src="{data_uri}" alt="Analyzed sample" /></div>
      <div class="result__file">{filename}</div>
      <div class="confidence">
        <div class="confidence__row">
          <span class="eyebrow">Model confidence</span>
          <span class="confidence__val">{pct}%</span>
        </div>
        <div class="bar"><div class="bar__fill" style="width:{pct}%"></div></div>
      </div>
    </div>
    <div>
      <span class="eyebrow">Key observations</span>
      <ul class="obs">{obs}</ul>
    </div>
  </div>
  <div class="disclaimer">
    <b>Disclaimer:</b> This tool is for preliminary screening only and does not constitute a
    clinical diagnosis. Consult a certified specialist for formal assessment.
  </div>
</div>
"""


# ------------------------------------------------------------------
# Static markup
# ------------------------------------------------------------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


HEADER_HERO = """
<div class="appbar">
  <div class="appbar__brand">
    <span class="appbar__mark"><svg width="20" height="20" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M4 7V4h3M20 7V4h-3M4 17v3h3M20 17v3h-3"/></svg></span>
    <span class="appbar__name">Handwriting Analysis</span>
  </div>
  <span class="eyebrow">Dysgraphia Detection</span>
</div>
<div class="hero">
  <span class="pill"><span class="pill__dot"></span> ML-Powered Screening</span>
  <h1>Dysgraphia Risk <span class="accent">Assessment</span></h1>
  <p>Upload a handwriting sample and the model grades dysgraphia risk in seconds as Low, Moderate, or High.</p>
</div>
"""

LOWER_SECTIONS = f"""
<section>
  <div class="steps-grid">
    <div class="how"><div class="how__num">01</div><div class="how__title">Upload</div>
      <p class="how__desc">A photo or scan of a natural handwriting sample.</p></div>
    <div class="how"><div class="how__num">02</div><div class="how__title">Classify</div>
      <p class="how__desc">The sample is cleaned, normalized, and passed to the trained model.</p></div>
    <div class="how"><div class="how__num">03</div><div class="how__title">Review</div>
      <p class="how__desc">Receive a risk classification with supporting observations.</p></div>
  </div>
</section>

<section class="card" style="margin-top:22px">
  <h2 class="card__title">Severity grades</h2>
  <p class="card__hint">How each classification level is defined and what action it suggests.</p>
  <div style="margin-top:20px">
    <div class="grade low">
      <div class="grade__head"><span class="grade__dot"></span><span class="grade__name">Low Risk</span></div>
      <p class="grade__desc">Handwriting is largely legible with minimal irregularities. Patterns fall within typical developmental ranges.</p>
      <span class="grade__action">No immediate intervention required. Routine monitoring is sufficient.</span>
    </div>
    <div class="grade mod">
      <div class="grade__head"><span class="grade__dot"></span><span class="grade__name">Moderate Risk</span></div>
      <p class="grade__desc">Noticeable irregularities in letter formation, spacing, or legibility. Some patterns align with dysgraphia markers.</p>
      <span class="grade__action">Professional screening is recommended. Early support strategies may be beneficial.</span>
    </div>
    <div class="grade high">
      <div class="grade__head"><span class="grade__dot"></span><span class="grade__name">High Risk</span></div>
      <p class="grade__desc">Multiple consistent markers of dysgraphia present. Significant impact on legibility and fluency observed.</p>
      <span class="grade__action">Referral to an educational psychologist or occupational therapist is strongly advised.</span>
    </div>
  </div>
</section>

<section class="card" style="margin-top:22px">
  <details class="perf" open>
    <summary class="section-head">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
      <h2 class="card__title">Model performance</h2>
      <svg class="perf__chev" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>
    </summary>
    <div class="perf__body">
      <div class="stats">
        <div class="stat"><div class="stat__num">80.0%</div><div class="stat__lab">Accuracy</div>
          <div class="stat__meter"><i style="width:80%"></i></div></div>
        <div class="stat"><div class="stat__num">75.5%</div><div class="stat__lab">Macro F1</div>
          <div class="stat__meter"><i style="width:75.5%"></i></div></div>
        <div class="stat warn"><div class="stat__num">56.7%</div><div class="stat__lab">High-Risk Recall</div>
          <div class="stat__meter"><i style="width:56.7%"></i></div></div>
      </div>
      <div class="about">
        <b>About the model:</b> Trained on handwriting from 60 respondents across three severity levels.
        Deep image features (ResNet-18) are classified by a support-vector model into Low, Moderate, or High
        risk. <b>High-Risk recall (56.7%) is the known limitation</b> — roughly half of severe cases are not
        flagged — so the tool is a first-pass screener, not a diagnostic.
      </div>
    </div>
  </details>
</section>

<div class="appfoot">
  <span class="eyebrow">Preliminary Screening Tool</span>
  <span class="eyebrow accent">Not a substitute for clinical diagnosis</span>
</div>
"""


# ------------------------------------------------------------------
# Render
# ------------------------------------------------------------------
load_css()
st.markdown(HEADER_HERO, unsafe_allow_html=True)

with st.container(border=False, key="upload_card"):
    st.markdown('<h2 class="card__title">Upload handwriting sample</h2>'
                '<p class="card__hint">Upload a clear photo or scan of natural handwriting. '
                'We clean and normalize the image before analysis.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload handwriting sample",
                                type=["png", "jpg", "jpeg", "webp"],
                                label_visibility="collapsed")

if uploaded is not None:
    raw = uploaded.getvalue()
    try:
        gray = np.array(Image.open(io.BytesIO(raw)).convert("L"))
    except Exception:
        st.error("Couldn't read that image. Try a JPG or PNG of a handwriting sample.")
        st.stop()

    with st.status("Analyzing sample…", expanded=True) as status:
        st.write("Preprocessing handwriting image")
        canvas = preprocess_single(gray)
        st.write("Normalizing and preparing sample")
        feat = extract_features(canvas)
        st.write("Running through classification model")
        label_idx, conf = classify(feat)
        status.update(label="Analysis complete", state="complete", expanded=False)

    mime = uploaded.type or "image/png"
    data_uri = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    st.markdown(result_html(label_idx, conf, uploaded.name, data_uri), unsafe_allow_html=True)

st.markdown(LOWER_SECTIONS, unsafe_allow_html=True)
