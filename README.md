# Arabic Sentiment Compass

**Arabic sentiment analysis powered by a fine-tuned, Twitter-adapted AraBERT.**

An end-to-end Arabic NLP project: a reproducible research pipeline for classifying noisy Arabic social-media text into **Negative, Neutral, or Positive**, and an Arabic RTL **Streamlit app** for real-time and CSV batch inference.

**[Try the live app](https://arabic-sentiment-compasss.streamlit.app/)** | [Source code](https://github.com/AbdelrhmanAkl/Arabic-Sentiment-Compass) | [Model on Hugging Face](https://huggingface.co/AbdelrahmanAkl/arabic-sentiment-compass-arabert)

---

## At a Glance

| | |
| --- | --- |
| **Task** | 3-class Arabic sentiment classification |
| **Model** | Twitter-adapted AraBERT (`aubmindlab/bert-base-arabertv02-twitter`), fine-tuned |
| **Test accuracy** | **88.22 ± 0.19%** |
| **Test macro F1** | **87.73 ± 0.17%** |
| **Training data** | 13,915 examples from ASTD + ArSAS |
| **Interface** | Arabic RTL Streamlit app, single text and CSV batch |

Results are the mean ± standard deviation over three training seeds on a fixed test split (details below).

---

## The Application

### Single-text analysis

- Arabic RTL interface
- Real transformer inference (no mock results)
- Predicted class in Arabic and English
- Confidence score and probabilities for all three classes
- Visual **sentiment compass**: a gauge whose needle moves between negative and positive
- One-click example texts (positive, negative, neutral)
- Warning when confidence is below 60%
- Input validation and loading states
- Preprocessed text view for transparency

### CSV batch analysis

1. Upload a CSV file (UTF-8 or UTF-8-SIG)
2. Choose the text column
3. Run the model on every row
4. Review summary statistics: totals, class percentages, average confidence
5. Preview the first 50 rows
6. Download the full results

All original columns are kept, and these columns are added:

```text
processed_text
sentiment
sentiment_ar
confidence
negative_probability
neutral_probability
positive_probability
```

Rows with an empty text cell are labelled `Neutral` with zero confidence and zero probabilities.

### Design

Arabic-first RTL layout with a colourful gradient identity. Each sentiment keeps the same colour everywhere: green for positive, amber for neutral, red for negative. The interface is responsive and respects reduced-motion preferences.

---

## Model

| Item | Value |
| --- | --- |
| Base checkpoint | `aubmindlab/bert-base-arabertv02-twitter` |
| Deployed model | [`AbdelrahmanAkl/arabic-sentiment-compass-arabert`](https://huggingface.co/AbdelrahmanAkl/arabic-sentiment-compass-arabert) |
| Architecture | `AutoModelForSequenceClassification` |
| Classes | 3 |
| Max sequence length | 128 tokens |
| Primary metric | Macro F1 |

The app loads the tokenizer and classifier from the Hugging Face Model Hub. The model checkpoint is not stored in this repository because of its size. On first launch, the download can take a moment.

| ID | Sentiment | Arabic |
| --: | --- | --- |
| `0` | Negative | سلبي |
| `1` | Neutral | محايد |
| `2` | Positive | إيجابي |

### Example

```text
Input:      الخدمة ممتازة والتجربة كانت رائعة جدًا!
Prediction: Positive — إيجابي
```

---

## Prediction Pipeline

Both single-text and CSV inputs follow the same path:

```text
        Single text  /  CSV rows
                  │
                  ▼
          Input validation
                  │
                  ▼
        Arabic preprocessing
                  │
                  ▼
              Tokenizer
                  │
                  ▼
     Twitter-adapted AraBERT  →  logits
                  │
                  ▼
        Softmax  →  probabilities
                  │
                  ▼
     Argmax  →  Negative / Neutral / Positive
                  │
                  ▼
       Results card  /  CSV export
```

**Confidence** is the highest Softmax score for the predicted class. It is **not** a calibrated probability.

### Arabic preprocessing

The function `preprocess_arabic_tweet` reduces noise while keeping sentiment-bearing content (such as emojis). It applies:

- URL and mention normalization
- Arabic diacritic removal
- Tatweel removal
- Alef normalization
- `ى` → `ي` normalization
- Reduction of excessive character repetition
- Whitespace normalization

The same preprocessing is used in training and in the deployed app to avoid train/inference mismatch.

---

## Experimental Methodology

The research was done in two stages.

### Stage I — ASTD-centred development

Explored preprocessing, class weighting, emoji handling, character normalization, training duration, AraSarcasm augmentation, random oversampling, and encoder choice.

| Model | Accuracy | Macro F1 |
| --- | --: | --: |
| Selected light-preprocessed AraBERT | 73.80% | 71.49% |
| MARBERT (strongest ASTD-only encoder) | 75.60% | 73.34% |

MARBERT was not carried into Stage II because Stage II was designed as a controlled comparison inside the AraBERT family.

### Stage II — combined ASTD + ArSAS experiment

The final corpus combines filtered examples from **ASTD** and **ArSAS**, with **13,915 unique examples**. ArSAS examples were kept for the three target classes using a sentiment-confidence threshold of at least `0.75`.

| Sentiment | Samples |
| --- | --: |
| Negative | 5,880 |
| Neutral | 5,062 |
| Positive | 2,973 |

| Split | Samples |
| --- | --: |
| Training | 11,132 |
| Validation | 1,391 |
| Test | 1,392 |

---

## Final Results

Evaluated on the fixed Stage II test split.

| System | Accuracy | Macro F1 |
| --- | --: | --: |
| Character TF-IDF + Logistic Regression | 81.47% | 80.42% |
| Standard AraBERT | 87.36 ± 0.45% | 86.83 ± 0.52% |
| **Twitter-adapted AraBERT** (deployed) | **88.22 ± 0.19%** | **87.73 ± 0.17%** |
| Baseline + raw-logit ensemble | 88.36 ± 0.26% | 87.89 ± 0.30% |

The Twitter-adapted AraBERT is the model used by the app. The ensemble is a research extension, not the deployed model.

### Egyptian-focused diagnostic

Evaluation on the **322 ASTD examples** inside the unseen Stage II test split:

| Metric | Score |
| --- | --: |
| Accuracy | 70.39 ± 1.00% |
| Macro F1 | 67.19 ± 0.48% |

The main Stage II result should **not** be described as an exclusively Egyptian Arabic result. The corpus is multi-source, and this subset is reported separately to show performance on the available ASTD data.

---

## Reproducibility

| Setting | Value |
| --- | --- |
| Base checkpoint | `aubmindlab/bert-base-arabertv02-twitter` |
| Max sequence length | 128 |
| Learning rate | `3e-5` |
| Weight decay | `0.01` |
| Max epochs | 5 |
| Early-stopping patience | 2 |
| Batch size per device | 16 |
| Gradient accumulation | 2 (effective batch size 32) |
| Gradient clipping | 1.0 |
| Loss | Class-weighted focal loss (γ = 2) |
| Data split seed | 42 |
| Training seeds | 21, 42, 77 |
| Primary metric | Macro F1 |

The full workflow (data preparation, filtering, training, evaluation, multi-seed experiments, statistics, and plots) is documented in the notebook:

```text
notebooks/Stage_II_Combined_ASTD_ArSAS_AraBERT_SentimentCompass_ipynb.ipynb
```

---

## Project Structure

```text
arabic-sentiment-compass/
├── app.py                  # Streamlit application
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── ArSAS.txt
└── notebooks/
    └── Stage_II_Combined_ASTD_ArSAS_AraBERT_SentimentCompass_ipynb.ipynb
```

---

## Run Locally

**Requirements:** Python **3.11 or newer** (the preprocessing regex uses possessive quantifiers).

```bash
# 1. Clone
git clone https://github.com/AbdelrhmanAkl/Arabic-Sentiment-Compass.git
cd Arabic-Sentiment-Compass

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run
python -m streamlit run app.py
```

The app opens in your browser. The first prediction downloads the model from the Hugging Face Hub.

---

## Tech Stack

| Category | Technology |
| --- | --- |
| Language | Python |
| Model | AraBERT (Twitter-adapted) |
| Deep learning | PyTorch |
| NLP library | Hugging Face Transformers |
| App | Streamlit |
| Model hosting | Hugging Face Model Hub |
| Deployment | Streamlit Community Cloud |
| Research | Jupyter Notebook |

---

## Limitations

- **Dataset composition:** the Stage II corpus is multi-source, not Egyptian-only.
- **Dialect variation:** Arabic varies widely across dialects, regions, topics, and writing styles.
- **Dataset differences:** collection methods, annotation guidelines, topics, and dialect mix cannot be fully harmonized.
- **Confidence:** the displayed score is a raw Softmax value and is not calibrated.
- **Input length:** texts longer than 128 tokens are truncated, so only the beginning is analysed.
- **Sarcasm and implicit meaning:** sarcasm, irony, implicit sentiment, and context-dependent expressions can be misclassified. Review low-confidence results manually.

---

## Future Work

- Confidence calibration
- Faster inference (quantization)
- More extensive Egyptian Arabic evaluation
- Error-analysis dashboard
- API endpoint
- Hugging Face Spaces deployment
- Comparison with additional Arabic Twitter models
- Explainability and token-level sentiment interpretation

---

## What This Project Demonstrates

An end-to-end Arabic NLP workflow, from data to a deployed product:

```text
Data preparation → Arabic preprocessing → Transformer fine-tuning
→ Multi-seed evaluation → Model comparison → Diagnostic analysis
→ Production model → Streamlit app (real-time + batch inference)
```

**Skills:** Arabic NLP · sentiment analysis · transformer fine-tuning · AraBERT · Hugging Face Transformers · PyTorch · class-imbalance handling · focal loss · multi-seed evaluation · macro-F1 benchmarking · Streamlit · model deployment · batch inference

---

## Links

- **Source code:** [GitHub](https://github.com/AbdelrhmanAkl/Arabic-Sentiment-Compass)
- **Live app:** [arabic-sentiment-compasss.streamlit.app](https://arabic-sentiment-compasss.streamlit.app/)
- **Model:** [Hugging Face](https://huggingface.co/AbdelrahmanAkl/arabic-sentiment-compass-arabert)

---

## License

This repository contains project code and experimental artifacts. Datasets and pretrained models remain subject to the licenses and terms of their original resources.

## Acknowledgments

Built on public Arabic NLP resources: **AraBERT**, **ASTD**, **ArSAS**, and **Hugging Face Transformers**.

---

**Built by [Abdelrahman Akl](https://github.com/AbdelrhmanAkl)** as an end-to-end Arabic NLP research and deployment project.
