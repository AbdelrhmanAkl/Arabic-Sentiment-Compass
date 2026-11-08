import re
import csv
import io
import math
from collections import Counter

import torch
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# =========================================================
# CONFIGURATION — MODEL PIPELINE UNCHANGED
# =========================================================

MODEL_ID = "AbdelrahmanAkl/arabic-sentiment-compass-arabert"
MAX_LENGTH = 128
GITHUB_URL = "https://github.com/AbdelrhmanAkl/Arabic-Sentiment-Compass"

ID2LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

ARABIC_LABELS = {
    "Negative": "سلبي",
    "Neutral": "محايد",
    "Positive": "إيجابي",
}

LABELS_ORDER = [
    "Positive",
    "Neutral",
    "Negative",
]

# Each sentiment keeps the same colour everywhere in the app.
SENTIMENT_COLORS = {
    "Negative": "#F43F5E",
    "Neutral": "#F59E0B",
    "Positive": "#10B981",
}

# Darker shade for text and a soft tint for backgrounds (readable contrast).
SENTIMENT_TEXT = {
    "Negative": "#BE123C",
    "Neutral": "#B45309",
    "Positive": "#047857",
}

SENTIMENT_TINT = {
    "Negative": "#FFE4E9",
    "Neutral": "#FEF1D2",
    "Positive": "#D5F7E8",
}

EXAMPLES = {
    "مثال إيجابي": "الخدمة كانت ممتازة والتجربة أفضل مما توقعت",
    "مثال سلبي": "تجربة سيئة جدًا، التوصيل اتأخر والدعم ما رد علي",
    "مثال محايد": "تم افتتاح الفرع الجديد يوم الأحد الساعة التاسعة صباحًا",
}

LOW_CONFIDENCE = 0.60


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="بوصلة المشاعر العربية | Arabic Sentiment Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# ORIGINAL PREPROCESSING — DO NOT CHANGE
# =========================================================

ARABIC_DIACRITICS = re.compile(
    r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]"
)

URL_RE = re.compile(r"https?://*\S*+|www\\.*\S*+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")


def preprocess_arabic_tweet(text):
    text = str(text)
    text = URL_RE.sub(" رابط ", text)
    text = MENTION_RE.sub(" مستخدم ", text)
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"ـ+", "", text)
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================================================
# MODEL LOADING — REAL MODEL ONLY
# =========================================================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
    )

    model.eval()

    return tokenizer, model


# =========================================================
# OFFICIAL PREDICTION PIPELINE — DO NOT CHANGE
# =========================================================

def predict_sentiment(text, tokenizer, model):

    processed_text = preprocess_arabic_tweet(text)

    encoded = tokenizer(
        processed_text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.inference_mode():

        logits = model(**encoded).logits

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

    scores = probabilities[0]

    predicted_id = int(scores.argmax())

    predicted_label = ID2LABEL[predicted_id]

    confidence = float(
        scores[predicted_id]
    )

    probabilities_dict = {
        ID2LABEL[i]: float(scores[i])
        for i in range(len(ID2LABEL))
    }

    return (
        processed_text,
        predicted_label,
        confidence,
        probabilities_dict,
    )


# =========================================================
# SESSION STATE
# =========================================================

if "single_result" not in st.session_state:
    st.session_state.single_result = None

if "csv_results" not in st.session_state:
    st.session_state.csv_results = None

if "csv_columns" not in st.session_state:
    st.session_state.csv_columns = []

if "single_text" not in st.session_state:
    st.session_state.single_text = ""


# =========================================================
# CALLBACKS
# =========================================================

def clear_single_input():
    """Clear the single-text widget and its latest prediction safely."""
    st.session_state["single_text"] = ""
    st.session_state["single_result"] = None


def set_example(text):
    st.session_state["single_text"] = text
    st.session_state["single_result"] = None


# =========================================================
# UI HELPERS
# =========================================================

def html(markup):
    """Render HTML without Markdown turning indented lines into code blocks."""
    st.markdown(re.sub(r"\n\s*", "", markup), unsafe_allow_html=True)


def _arc(cx, cy, r, a0, a1):
    x0 = cx + r * math.cos(math.radians(a0))
    y0 = cy - r * math.sin(math.radians(a0))
    x1 = cx + r * math.cos(math.radians(a1))
    y1 = cy - r * math.sin(math.radians(a1))
    return f"M {x0:.2f} {y0:.2f} A {r} {r} 0 0 1 {x1:.2f} {y1:.2f}"


def gauge_svg(label=None, score=0.0, on_dark=False):
    """
    Semicircular compass gauge.
    Left = negative, centre = neutral, right = positive.
    score = P(positive) - P(negative), range [-1, 1].
    """
    cx, cy, r = 150, 150, 118
    idle = label is None
    grey = "#3A4A63" if on_dark else "#E3E7EE"
    segments = [
        ("Negative", 178.5, 121.5),
        ("Neutral", 118.5, 61.5),
        ("Positive", 58.5, 1.5),
    ]
    arcs = ""
    for name, a0, a1 in segments:
        if idle:
            color = "#FFFFFF" if on_dark else "#CDD1F2"
            opacity = 0.45 if on_dark else 1
        else:
            color = SENTIMENT_COLORS[name]
            opacity = 1 if name == label else 0.22
        arcs += (
            f'<path d="{_arc(cx, cy, r, a0, a1)}" fill="none" stroke="{color}" '
            f'stroke-width="20" stroke-opacity="{opacity}"/>'
        )

    needle_color = "#FFFFFF" if on_dark else "#0E1A2B"
    deg = max(-1.0, min(1.0, score)) * 90
    needle = (
        f'<g class="needle" style="--r:{deg:.1f}deg;transform:rotate({deg:.1f}deg);'
        f'transform-origin:{cx}px {cy}px;">'
        f'<path d="M {cx-5} {cy} L {cx} {cy-r+26} L {cx+5} {cy} Z" fill="{needle_color}"/>'
        f"</g>"
        f'<circle cx="{cx}" cy="{cy}" r="9" fill="{needle_color}"/>'
        f'<circle cx="{cx}" cy="{cy}" r="3.5" fill="{"#0E1A2B" if on_dark else "#FFFFFF"}"/>'
    )

    tick_color = "#F1EEFF" if on_dark else "#6B7788"
    ticks = (
        f'<text x="{cx-r}" y="{cy+30}" text-anchor="middle" font-size="13" fill="{tick_color}">سلبي</text>'
        f'<text x="{cx}" y="{cy-r-16}" text-anchor="middle" font-size="13" fill="{tick_color}">محايد</text>'
        f'<text x="{cx+r}" y="{cy+30}" text-anchor="middle" font-size="13" fill="{tick_color}">إيجابي</text>'
    )

    return (
        f'<svg viewBox="0 0 300 190" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="مؤشر اتجاه المشاعر" style="width:100%;height:auto;display:block;">'
        f"{arcs}{needle}{ticks}</svg>"
    )


# =========================================================
# STYLES
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=Readex+Pro:wght@400;500;600;700&display=swap');

/* One brand colour (indigo). Green / amber / red appear only on sentiment results. */
:root{
    --ink:#1E1B4B;
    --ink-2:#3F3C70;
    --muted:#65688F;
    --line:#DDE0F5;
    --surface:#FFFFFF;
    --tint:#EEF0FF;
    --brand:#4338CA;
    --brand-dark:#3730A3;
    --neg:#BE123C;
    --neu:#B45309;
    --pos:#047857;
}

html, body, [class*="css"], .stApp, button, input, textarea{
    font-family:"IBM Plex Sans Arabic","Readex Pro",sans-serif;
}

.stApp{ direction:rtl; color:var(--ink); background:#F5F6FF; }
.block-container{ max-width:1180px; padding:1.4rem 2rem 3rem; }

#MainMenu, footer, header{ visibility:hidden; }
[data-testid="stToolbar"], [data-testid="stDecoration"]{ display:none; }
[data-testid="stDataFrame"], [data-testid="stCode"], pre, code{ direction:ltr; text-align:left; }

/* ---------- top bar ---------- */
.topbar{ display:flex; align-items:center; justify-content:space-between; padding:6px 2px 18px; }
.brand{ display:flex; align-items:center; gap:12px; }
.brand-mark{
    width:44px; height:44px; border-radius:13px; background:var(--brand);
    display:flex; align-items:center; justify-content:center;
}
.brand-name{ font-family:"Readex Pro",sans-serif; font-weight:700; font-size:17px; color:var(--ink); line-height:1.2; }
.brand-sub{ font-size:13px; color:var(--muted); }
.gh-link{
    font-size:14px; font-weight:600; color:var(--brand) !important; text-decoration:none !important;
    background:#fff; border:1px solid var(--line); padding:10px 18px; border-radius:12px;
}
.gh-link:hover{ border-color:var(--brand); }

/* ---------- hero ---------- */
.hero{
    color:#fff; border-radius:24px; padding:46px 50px;
    background:linear-gradient(135deg,#312E81 0%,#4338CA 100%);
    display:grid; grid-template-columns:1.2fr .8fr; gap:40px; align-items:center;
    margin-bottom:28px;
}
.hero h1{
    font-family:"Readex Pro",sans-serif; font-weight:700;
    font-size:clamp(30px,4.2vw,44px); line-height:1.35; margin:0 0 14px; color:#fff;
}
.hero p{ font-size:17px; line-height:1.95; color:rgba(255,255,255,.88); margin:0 0 28px; max-width:540px; }
.facts{ display:flex; margin:0; padding:0; }
.fact{ padding:0 22px; border-inline-start:1px solid rgba(255,255,255,.28); }
.fact:first-child{ padding-inline-start:0; border-inline-start:0; }
.fact b{ display:block; font-family:"Readex Pro",sans-serif; font-weight:700; font-size:19px; color:#fff; }
.fact span{ font-size:13px; color:rgba(255,255,255,.78); }
.hero-gauge{
    direction:ltr; width:100%; max-width:340px; margin-inline:auto;
    background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.22);
    border-radius:20px; padding:22px 20px 12px;
}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"]{ gap:8px; border-bottom:1px solid var(--line); padding:0; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none; }
.stTabs [data-baseweb="tab"]{
    height:46px; padding:0 20px; font-size:15px; font-weight:600; color:var(--muted);
    background:transparent; border-radius:10px 10px 0 0; border-bottom:3px solid transparent;
}
.stTabs [aria-selected="true"]{ color:var(--brand) !important; background:#fff; border-bottom-color:var(--brand); }
.stTabs [aria-selected="true"] p{ color:var(--brand) !important; }

/* ---------- headings ---------- */
.h2{ font-family:"Readex Pro",sans-serif; font-weight:700; font-size:24px; color:var(--ink); margin:22px 0 6px; }
.lead{ font-size:15px; color:var(--muted); line-height:1.9; margin:0 0 18px; }
.field-label{ font-size:15px; font-weight:600; color:var(--ink); margin:4px 0 8px; }
.hint{ font-size:13px; color:var(--muted); }

/* ---------- text input ---------- */
[data-testid="stTextArea"] textarea{
    direction:rtl; text-align:right;
    font-size:17px !important; line-height:1.9 !important; color:var(--ink) !important;
    background:#fff !important; border:1px solid var(--line) !important; border-radius:14px !important;
    min-height:200px !important; padding:16px 18px !important;
}
[data-testid="stTextArea"] textarea:focus{
    border-color:var(--brand) !important; box-shadow:0 0 0 3px rgba(67,56,202,.15) !important;
}

/* ---------- buttons ---------- */
.stButton > button{
    min-height:48px; border-radius:12px; font-weight:600; font-size:15px;
    background:#fff; color:var(--ink); border:1px solid var(--line); box-shadow:none;
    transition:border-color .15s ease, background .15s ease;
}
.stButton > button:hover{ border-color:var(--brand); color:var(--brand); background:#fff; }
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"]{ background:var(--brand); color:#fff; border:1px solid var(--brand); }
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover{ background:var(--brand-dark); border-color:var(--brand-dark); color:#fff; }
.stButton > button:focus-visible{ outline:3px solid rgba(67,56,202,.35); outline-offset:2px; }
[data-testid="stDownloadButton"] > button{
    min-height:48px; border-radius:12px; font-weight:600; font-size:15px;
    color:#fff; background:var(--brand); border:1px solid var(--brand);
}
[data-testid="stDownloadButton"] > button:hover{ background:var(--brand-dark); color:#fff; }

/* ---------- result panel ---------- */
.panel{ background:#fff; border:1px solid var(--line); border-radius:20px; padding:26px 28px; }
.panel-idle{ min-height:420px; display:flex; flex-direction:column; justify-content:center; background:var(--tint); border-color:transparent; }
.panel-result{ border-top:6px solid var(--accent); }
.result-top{ display:flex; align-items:center; justify-content:space-between; gap:14px; }
.result-kicker{ font-size:14px; color:var(--muted); font-weight:500; }
.badge{ font-size:13px; font-weight:700; padding:6px 14px; border-radius:999px; background:var(--tint); color:var(--brand); }
.result-label{ font-family:"Readex Pro",sans-serif; font-weight:700; font-size:46px; line-height:1.2; margin:6px 0 0; }
.result-sub{ font-size:14px; color:var(--muted); margin-top:2px; }
.gauge-box{ direction:ltr; max-width:360px; margin:14px auto 6px; }
.needle{ animation:sweep .9s cubic-bezier(.2,.8,.2,1); }
@keyframes sweep{ from{ transform:rotate(0deg); } }
@media (prefers-reduced-motion:reduce){ .needle{ animation:none; } }

.probs{ margin-top:12px; border-top:1px solid var(--line); padding-top:16px; }
.prob{ display:grid; grid-template-columns:70px 1fr 64px; gap:12px; align-items:center; margin-bottom:12px; }
.prob-name{ font-size:15px; font-weight:600; color:var(--ink); }
.prob-val{ font-size:15px; font-weight:700; color:var(--ink); text-align:left; direction:ltr; font-variant-numeric:tabular-nums; }
.track{ height:10px; border-radius:99px; background:var(--tint); overflow:hidden; direction:rtl; }
.fill{ height:100%; border-radius:99px; }

.notice{
    margin-top:12px; padding:12px 16px; border-radius:12px; font-size:14px; line-height:1.8;
    background:var(--tint); border:1px solid var(--line); color:var(--ink-2);
}

/* ---------- stats / batch ---------- */
.stats{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:14px; }
.stat{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:16px 18px; border-top:4px solid var(--line); }
.stat-k{ font-size:14px; font-weight:600; color:var(--muted); }
.stat-v{ font-family:"Readex Pro",sans-serif; font-size:30px; font-weight:700; line-height:1.3; margin-top:2px; color:var(--ink); }
.stat-s{ font-size:13px; color:var(--muted); }
.s-total{ background:var(--brand); border-color:var(--brand); }
.s-total .stat-k, .s-total .stat-s{ color:rgba(255,255,255,.82); }
.s-total .stat-v{ color:#fff; }
.s-pos{ border-top-color:#10B981; } .s-pos .stat-v{ color:var(--pos); }
.s-neu{ border-top-color:#F59E0B; } .s-neu .stat-v{ color:var(--neu); }
.s-neg{ border-top-color:#F43F5E; } .s-neg .stat-v{ color:var(--neg); }
.stack{ display:flex; height:14px; border-radius:99px; overflow:hidden; direction:rtl; margin:4px 0 18px; background:var(--tint); }

[data-testid="stFileUploader"] section{ border:1.5px dashed #AEB4EC; border-radius:14px; background:#fff; }
[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:12px; overflow:hidden; }

/* ---------- about ---------- */
.kv{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:8px 0; }
.kv > div{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px 18px; }
.kv .k{ font-size:13px; color:var(--muted); }
.kv .v{ font-family:"Readex Pro",sans-serif; font-weight:700; font-size:17px; color:var(--ink); margin-top:3px; word-break:break-all; direction:ltr; text-align:right; }
.steps{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; counter-reset:s; margin:8px 0; }
.steps > div{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px; counter-increment:s; }
.steps > div:before{
    content:counter(s); display:flex; align-items:center; justify-content:center;
    width:30px; height:30px; border-radius:50%; background:var(--brand); color:#fff;
    font-size:14px; font-weight:700; margin-bottom:10px;
}
.steps b{ display:block; font-size:15px; color:var(--ink); }
.steps span{ font-size:13px; color:var(--muted); line-height:1.7; }
.limits{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:18px 24px; font-size:15px; line-height:2; color:var(--ink-2); }
.limits ul{ margin:0; padding-inline-start:20px; }

.footer{
    margin-top:42px; padding-top:18px; border-top:1px solid var(--line);
    display:flex; justify-content:space-between; gap:14px; font-size:13px; color:var(--muted);
}
.footer a{ color:var(--brand); font-weight:700; text-decoration:none; }

@media(max-width:900px){
    .block-container{ padding:1rem 1rem 2rem; }
    .hero{ grid-template-columns:1fr; padding:30px 24px; gap:24px; }
    .stats, .kv{ grid-template-columns:1fr 1fr; }
    .steps{ grid-template-columns:1fr; }
    .facts{ flex-wrap:wrap; row-gap:14px; }
    .footer{ flex-direction:column; }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# TOP BAR + HERO
# =========================================================

html(f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-mark"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M15.5 8.5l-2 5-5 2 2-5z" fill="#fff"/></svg></div>
    <div>
      <div class="brand-name">بوصلة المشاعر العربية</div>
      <div class="brand-sub">Arabic Sentiment Compass</div>
    </div>
  </div>
  <a class="gh-link" href="{GITHUB_URL}" target="_blank" rel="noopener">الكود على GitHub</a>
</div>

<div class="hero">
  <div>
    <h1>اعرف اتجاه المشاعر في أي نص عربي خلال ثوانٍ</h1>
    <p>الصق تعليقًا أو تغريدة أو رأيًا، أو ارفع ملف CSV كاملًا. يصنّف النموذج كل نص إلى إيجابي أو سلبي أو محايد، ويعرض نسبة الثقة وتوزيع الاحتمالات.</p>
    <div class="facts">
      <div class="fact"><b>AraBERT</b><span>نموذج لغوي للعربية</span></div>
      <div class="fact"><b>3 فئات</b><span>سلبي، محايد، إيجابي</span></div>
      <div class="fact"><b>CSV</b><span>تحليل ملفات كاملة</span></div>
    </div>
  </div>
  <div class="hero-gauge">{gauge_svg(None, 0, on_dark=True)}</div>
</div>
""")


# =========================================================
# TABS
# =========================================================

tab_text, tab_csv, tab_about = st.tabs(
    ["تحليل نص", "تحليل ملف CSV", "عن المشروع"]
)


# =========================================================
# TAB 1 — SINGLE TEXT
# =========================================================
with tab_text:
    result = st.session_state.get("single_result")

    left, right = st.columns([1, 1], gap="large")

    with left:
        html('<div class="h2">أدخل النص</div><p class="lead">اكتب جملة أو تغريدة أو رأيًا بالعربية، أو جرّب أحد الأمثلة.</p>')

        ex_cols = st.columns(len(EXAMPLES), gap="small")
        for i, (col, (name, sample)) in enumerate(zip(ex_cols, EXAMPLES.items())):
            with col:
                st.button(
                    name,
                    key=f"ex_{i}",
                    use_container_width=True,
                    on_click=set_example,
                    args=(sample,),
                )

        user_text = st.text_area(
            "النص العربي",
            placeholder="مثال: الخدمة كانت ممتازة والتجربة أفضل مما توقعت",
            height=200,
            label_visibility="collapsed",
            key="single_text",
        )

        b1, b2 = st.columns([2, 1], gap="small")
        with b1:
            analyze_button = st.button(
                "حلّل المشاعر",
                type="primary",
                use_container_width=True,
                key="analyze_single",
            )
        with b2:
            st.button(
                "مسح",
                use_container_width=True,
                key="clear_single",
                on_click=clear_single_input,
            )

        if analyze_button:
            if not user_text.strip():
                st.warning("اكتب نصًا عربيًا أولًا ثم اضغط «حلّل المشاعر».")
            else:
                try:
                    with st.spinner("جارٍ تحليل النص..."):
                        tokenizer, model = load_model()
                        processed_text, predicted_label, confidence, probabilities_dict = predict_sentiment(
                            user_text, tokenizer, model
                        )
                    st.session_state.single_result = {
                        "processed_text": processed_text,
                        "predicted_label": predicted_label,
                        "confidence": confidence,
                        "probabilities_dict": probabilities_dict,
                    }
                    result = st.session_state.single_result
                except Exception as exc:
                    st.error("حدث خطأ أثناء تشغيل النموذج.")
                    st.exception(exc)

    with right:
        html('<div class="h2">النتيجة</div><p class="lead">القرار النهائي ودرجة الثقة وتوزيع الاحتمالات.</p>')

        if not result:
            html(f"""
            <div class="panel panel-idle">
              <div class="gauge-box">{gauge_svg(None, 0)}</div>
              <div style="text-align:center;margin-top:10px;">
                <div style="font-family:'Readex Pro',sans-serif;font-weight:600;font-size:20px;color:#0E1A2B;">لا توجد نتيجة بعد</div>
                <div class="hint" style="margin-top:4px;">اكتب نصًا واضغط «حلّل المشاعر» لتظهر النتيجة هنا.</div>
              </div>
            </div>
            """)
        else:
            label = result["predicted_label"]
            probs = result["probabilities_dict"]
            conf = result["confidence"]
            score = probs["Positive"] - probs["Negative"]
            color = SENTIMENT_COLORS[label]
            tcolor = SENTIMENT_TEXT[label]
            tint = SENTIMENT_TINT[label]

            rows = ""
            for name in LABELS_ORDER:
                v = probs[name] * 100
                rows += (
                    f'<div class="prob"><div class="prob-name">{ARABIC_LABELS[name]}</div>'
                    f'<div class="track"><div class="fill" style="width:{max(0, min(100, v)):.2f}%;background:{SENTIMENT_COLORS[name]};"></div></div>'
                    f'<div class="prob-val">{v:.2f}%</div></div>'
                )

            html(f"""
            <div class="panel panel-result" style="--accent:{color};">
              <div class="result-top">
                <div class="result-kicker">التصنيف المتوقع</div>
                <div class="badge">الثقة {conf*100:.1f}%</div>
              </div>
              <div class="result-label" style="color:{tcolor};">{ARABIC_LABELS[label]}</div>
              <div class="result-sub">{label}</div>
              <div class="gauge-box">{gauge_svg(label, score)}</div>
              <div class="probs">{rows}</div>
            </div>
            """)

            if conf < LOW_CONFIDENCE:
                html('<div class="notice">درجة الثقة منخفضة، والاحتمالات متقاربة بين الفئات. يُفضّل مراجعة هذا النص يدويًا.</div>')

            with st.expander("النص بعد المعالجة (preprocessing)"):
                st.code(result["processed_text"], language="text")


# =========================================================
# TAB 2 — CSV
# =========================================================
with tab_csv:
    html('<div class="h2">حلّل ملفًا كاملًا</div><p class="lead">ارفع ملف CSV، اختر عمود النص، وسيُضاف لكل صف التصنيف والثقة والاحتمالات.</p>')

    left, right = st.columns([.9, 1.1], gap="large")

    with left:
        html('<div class="field-label">ملف البيانات</div><div class="hint" style="margin-bottom:8px;">ترميز UTF-8 أو UTF-8-SIG</div>')
        uploaded_file = st.file_uploader(
            "رفع ملف CSV",
            type=["csv"],
            key="csv_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.getvalue()
                try:
                    csv_text = raw_bytes.decode("utf-8-sig")
                except UnicodeDecodeError:
                    csv_text = raw_bytes.decode("utf-8")

                reader = csv.DictReader(io.StringIO(csv_text))
                rows = list(reader)
                fieldnames = reader.fieldnames or []

                if not rows or not fieldnames:
                    st.warning("الملف فارغ أو لا يحتوي على أعمدة.")
                else:
                    text_column = st.selectbox(
                        "عمود النص",
                        fieldnames,
                        key="csv_text_column",
                    )
                    st.caption(f"عدد السجلات: {len(rows):,}")

                    analyze_csv_button = st.button(
                        "ابدأ تحليل الملف",
                        type="primary",
                        use_container_width=True,
                        key="analyze_csv",
                    )

                    if analyze_csv_button:
                        try:
                            with st.spinner("جارٍ تحليل الملف..."):
                                tokenizer, model = load_model()
                                results = []
                                progress = st.progress(0.0)
                                total = len(rows)

                                for index, row in enumerate(rows, start=1):
                                    text_value = str(row.get(text_column, "") or "")
                                    if text_value.strip():
                                        processed_text, predicted_label, confidence, probabilities_dict = predict_sentiment(
                                            text_value, tokenizer, model
                                        )
                                    else:
                                        processed_text = ""
                                        predicted_label = "Neutral"
                                        confidence = 0.0
                                        probabilities_dict = {
                                            "Negative": 0.0,
                                            "Neutral": 0.0,
                                            "Positive": 0.0,
                                        }

                                    new_row = dict(row)
                                    new_row["processed_text"] = processed_text
                                    new_row["sentiment"] = predicted_label
                                    new_row["sentiment_ar"] = ARABIC_LABELS[predicted_label]
                                    new_row["confidence"] = round(confidence, 6)
                                    new_row["negative_probability"] = round(probabilities_dict["Negative"], 6)
                                    new_row["neutral_probability"] = round(probabilities_dict["Neutral"], 6)
                                    new_row["positive_probability"] = round(probabilities_dict["Positive"], 6)
                                    results.append(new_row)
                                    progress.progress(index / total)

                            st.session_state.csv_results = results
                            st.session_state.csv_columns = list(results[0].keys()) if results else []
                            st.success("تم تحليل الملف بنجاح.")
                        except Exception as exc:
                            st.error("حدث خطأ أثناء تحليل ملف CSV.")
                            st.exception(exc)
            except Exception as exc:
                st.error("تعذر قراءة ملف CSV.")
                st.exception(exc)

    with right:
        csv_results = st.session_state.get("csv_results")

        if not csv_results:
            html("""
            <div class="panel" style="min-height:300px;display:flex;flex-direction:column;justify-content:center;text-align:center;">
              <div style="font-family:'Readex Pro',sans-serif;font-weight:600;font-size:20px;color:#0E1A2B;">لا توجد نتائج بعد</div>
              <div class="hint" style="margin-top:6px;">ارفع ملف CSV واضغط «ابدأ تحليل الملف» لتظهر الإحصائيات هنا.</div>
            </div>
            """)
        else:
            counts = Counter(row.get("sentiment", "Neutral") for row in csv_results)
            total_rows = len(csv_results)
            analysed = [r for r in csv_results if r.get("processed_text")]
            avg_conf = (
                sum(float(r["confidence"]) for r in analysed) / len(analysed) * 100
                if analysed else 0.0
            )

            def pct(name):
                return counts.get(name, 0) / total_rows * 100

            html(f"""
            <div class="stats">
              <div class="stat s-total"><div class="stat-k">إجمالي السجلات</div><div class="stat-v">{total_rows:,}</div><div class="stat-s">متوسط الثقة {avg_conf:.1f}%</div></div>
              <div class="stat s-pos"><div class="stat-k">إيجابي</div><div class="stat-v">{pct('Positive'):.1f}%</div><div class="stat-s">{counts.get('Positive',0):,} سجل</div></div>
              <div class="stat s-neu"><div class="stat-k">محايد</div><div class="stat-v">{pct('Neutral'):.1f}%</div><div class="stat-s">{counts.get('Neutral',0):,} سجل</div></div>
              <div class="stat s-neg"><div class="stat-k">سلبي</div><div class="stat-v">{pct('Negative'):.1f}%</div><div class="stat-s">{counts.get('Negative',0):,} سجل</div></div>
            </div>
            <div class="stack">
              <div style="width:{pct('Positive'):.2f}%;background:var(--pos);"></div>
              <div style="width:{pct('Neutral'):.2f}%;background:var(--neu);"></div>
              <div style="width:{pct('Negative'):.2f}%;background:var(--neg);"></div>
            </div>
            """)

            html('<div class="field-label">معاينة أول 50 صفًا</div>')
            st.dataframe(csv_results[:50], use_container_width=True, hide_index=True)

            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=st.session_state.get("csv_columns", []),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(csv_results)

            st.download_button(
                "تنزيل النتائج كاملة (CSV)",
                data=output.getvalue().encode("utf-8-sig"),
                file_name="arabic_sentiment_results.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_csv_results",
            )


# =========================================================
# TAB 3 — ABOUT
# =========================================================
with tab_about:
    html(f"""
    <div class="h2">عن المشروع</div>
    <p class="lead">نظام لتحليل المشاعر في النصوص العربية، يعمل بنموذج AraBERT حقيقي بدون نتائج تجريبية أو ثابتة. يدعم تحليل نص واحد أو ملف CSV كامل.</p>

    <div class="kv">
      <div><div class="k">النموذج</div><div class="v">{MODEL_ID.split('/')[-1]}</div></div>
      <div><div class="k">الفئات</div><div class="v" style="direction:rtl;">سلبي، محايد، إيجابي</div></div>
      <div><div class="k">الحد الأقصى للنص</div><div class="v">{MAX_LENGTH} tokens</div></div>
    </div>

    <div class="h2">كيف يعمل التحليل</div>
    <div class="steps">
      <div><b>التحقق</b><span>التأكد من أن النص غير فارغ</span></div>
      <div><b>التنظيف</b><span>توحيد الحروف وإزالة التشكيل واستبدال الروابط والمنشنات</span></div>
      <div><b>التقطيع</b><span>تحويل النص إلى tokens بواسطة tokenizer الخاص بـ AraBERT</span></div>
      <div><b>النموذج</b><span>حساب درجات الفئات الثلاث</span></div>
      <div><b>القرار</b><span>Softmax ثم اختيار الفئة الأعلى احتمالًا</span></div>
    </div>

    <div class="h2">حدود الاستخدام</div>
    <div class="limits">
      <ul>
        <li>النموذج مُدرَّب على نصوص قصيرة شبيهة بالتغريدات، ويكون أدق مع هذا النوع من النصوص.</li>
        <li>النصوص الأطول من {MAX_LENGTH} رمزًا تُقتطع، فيُحلَّل أولها فقط.</li>
        <li>السخرية والمعاني الضمنية قد تُصنَّف خطأً، لذلك راجع النتائج منخفضة الثقة يدويًا.</li>
      </ul>
    </div>
    """)


# =========================================================
# FOOTER
# =========================================================

html(f"""
<div class="footer">
  <div>Arabic Sentiment Compass · تطوير <a href="{GITHUB_URL}" target="_blank" rel="noopener">عبدالرحمن عقل</a></div>
  <div>Streamlit · Hugging Face Transformers · AraBERT</div>
</div>
""")
