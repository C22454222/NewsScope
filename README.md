# NewsScope — Bias-Aware News Aggregator with AI Analysis

[![Backend CI](https://github.com/C22454222/NewsScope/actions/workflows/backend-ci.yml/badge.svg)](https://github.com/C22454222/NewsScope/actions/workflows/backend-ci.yml)
[![Frontend CI](https://github.com/C22454222/NewsScope/actions/workflows/frontend-ci.yml/badge.svg)](https://github.com/C22454222/NewsScope/actions/workflows/frontend-ci.yml)
[![CodeQL](https://github.com/C22454222/NewsScope/actions/workflows/codeql.yml/badge.svg)](https://github.com/C22454222/NewsScope/actions/workflows/codeql.yml)

> **Final Year Project — Technological University Dublin**
> BSc in Computer Science (TU856), 2025–2026
> Author: Christopher Noblett (C22454222)
> Supervisor: Eoin Rogers

NewsScope is a free, mobile-first Android application that addresses declining
public trust in news media by analysing individual articles for political bias,
emotional sentiment, and factual credibility using transformer-based AI. Unlike
commercial platforms such as Ground News and AllSides — which restrict
meaningful features to paid subscriptions and classify entire outlets rather
than individual articles — NewsScope delivers article-level analysis to a
consumer-facing audience at no cost.

**Live backend API (Swagger):** [Swagger UI](https://newsscope-backend.onrender.com/docs)

---

## Key Features

- **Hourly automated ingestion** from twelve international news sources via
  NewsAPI and RSS, with source-specific scrapers for each outlet, URL-based
  deduplication, and inline category inference at ingestion time.
- **Article-level political bias classification** using a fine-tuned RoBERTa
  model trained on the Baly et al. Article-Bias-Prediction dataset, classifying
  articles as Left, Centre, or Right with a confidence score (87% test accuracy).
- **Sentence-level general bias detection** using a DistilRoBERTa model
  fine-tuned on the BABE dataset (Spinde et al., 2021), classifying content as
  biased or unbiased (83% test accuracy, +14 points over the off-the-shelf
  baseline).
- **News-domain sentiment analysis** using DistilBERT-SST-2 mapped to a
  continuous [-1, +1] scale, benchmarked at 91% binary accuracy on the
  FinancialPhraseBank corpus.
- **LIME word-level explainability** for high-confidence bias classifications,
  surfacing the words most responsible for each prediction so users can see
  exactly why the model labelled an article the way it did.
- **Credibility scoring** integrating the Google Fact Check Tools API with
  source reputation weighting, producing a 0–100 credibility score and
  structured fact-check rows per article.
- **Side-by-side story comparison** screen grouping articles on the same topic
  by political leaning into Left/Centre/Right tabs.
- **Personalised bias profile** with time-weighted reading history, donut chart
  visualisations of political leaning distribution, source breakdown bar chart,
  and average credibility tracking.
- **Push notifications** via Firebase Cloud Messaging when new articles are
  ingested, with per-user opt-in via the settings screen.
- **GDPR-compliant data controls** including reading history clearance, full
  account deletion with re-authentication, and a privacy policy disclosure
  bottom sheet.
- **Dark mode**, display preference toggles, glossary of analytical terms, and
  Material Design 3 UI throughout.

---

## Architecture

NewsScope is a four-tier distributed system deployed entirely on free-tier
cloud infrastructure:

1. **Presentation tier** — Flutter Android client with Firebase Authentication,
   Provider state management, and `fl_chart` for visualisations.
2. **Application tier** — FastAPI backend on Render with APScheduler driving
   the hourly ingestion → analysis → fact-check chain. A redeploy guard, mutual
   exclusion lock, and chain-first scheduler keep the pipeline within the
   512 MB memory limit of the Render free tier.
3. **AI/NLP tier** — Three Hugging Face Spaces hosting the political bias,
   general bias, and sentiment models behind the Gradio SSE protocol. Models
   are loaded once on Space startup and cached.
4. **Data tier** — Supabase PostgreSQL with five tables (`articles`, `users`,
   `reading_history`, `sources`, `fact_checks`), Row Level Security policies
   blocking direct anon access, and a daily archiving job moving articles
   older than seven days to Supabase Storage as JSON.

A complete architectural overview, sequence diagrams, and database schema are
provided in Chapter 4 of the dissertation.

---

## Tech Stack

| Tier | Technologies |
| --- | --- |
| **Frontend** | Flutter (Dart 3.9+), Firebase Auth, Firebase Messaging, Provider, fl_chart, flutter_local_notifications |
| **Backend** | FastAPI, Python 3.11, APScheduler, Pydantic, httpx, BeautifulSoup, newspaper3k |
| **Data** | Supabase (PostgreSQL + Storage), Firebase Authentication |
| **AI/NLP** | Hugging Face Spaces (Gradio), RoBERTa, DistilRoBERTa, DistilBERT, LIME |
| **Infra** | Render (backend hosting), Hugging Face Spaces (model hosting), GitHub Actions (CI/CD) |
| **Security** | Firebase JWT validation, Supabase Row Level Security, GitHub CodeQL, Bandit static analysis |

---

## Models

NewsScope uses three transformer models, two fine-tuned by the project author
and one off-the-shelf:

| Model | Base | Dataset | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| **Political bias** ([C22454222/political-bias-roberta](https://huggingface.co/C22454222/political-bias-roberta)) | roberta-base | Baly et al. Article-Bias-Prediction | 87% | 0.87 |
| **General bias** ([C22454222/general-bias-distilroberta](https://huggingface.co/C22454222/general-bias-distilroberta)) | distilroberta-base | BABE (Spinde et al. 2021) | 83% | 0.83 |
| **Sentiment** ([distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)) | distilbert-base | SST-2 (off-the-shelf) | 91% (binary) | 0.89 |

Training notebooks and benchmarking scripts are in the `notebooks/` directory.
Full evaluation methodology, per-class metrics, confusion matrices, and
training curves are documented in Chapter 6 of the dissertation.

---

## Repository Structure

```text
NewsScope/
├── backend/                       FastAPI server
│   ├── app/
│   │   ├── core/                  Config, scheduler, categorisation
│   │   ├── db/                    Supabase client
│   │   ├── jobs/                  Ingestion, analysis, fact-checking, archiving
│   │   ├── routes/                articles, users, sources routers
│   │   ├── services/              LIME explainability service
│   │   ├── main.py                Application entry point + lifespan
│   │   └── schemas.py             Pydantic models
│   └── requirements.txt
├── frontend/newsscope/            Flutter Android application
│   ├── lib/
│   │   ├── core/                  App preferences, configuration
│   │   ├── models/                Article, BiasProfile data models
│   │   ├── screens/               Home, Compare, Profile, Settings, Auth, Detail
│   │   ├── services/              ApiService — backend client
│   │   ├── utils/                 Score helpers, formatters
│   │   └── widgets/               ArticleCard, BiasChip
│   └── pubspec.yaml
├── notebooks/                     Model training and benchmarking
│   ├── PoliticalBiasModel_v2.ipynb
│   ├── GeneralBiasModel.ipynb
│   └── SentimentBenchmark.ipynb
├── docs/                          Proposal, Interim Report, Final Report
├── .github/workflows/             CI/CD pipelines (backend, frontend, CodeQL, deploy)
├── README.md
└── SECURITY.md
```

---

## Local Setup

### Prerequisites

- Flutter SDK 3.9+ (`flutter doctor` to verify)
- Python 3.11
- Git
- A Supabase project, Firebase project, and the relevant API keys (see
  the dissertation submission for the credentials bundle, or provision your
  own — see `.env.example` in the backend directory)

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy .env.example to .env and populate with your keys
cp .env.example .env

# Run locally with hot reload
uvicorn app.main:app --reload
```

The backend exposes Swagger documentation at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend/newsscope
flutter pub get

# Connect an Android device or start an emulator, then:
flutter run
```

The Flutter client points at the production backend by default. To target
a local backend, override `BASE_URL` at build time:

```bash
flutter run --dart-define=BASE_URL=http://10.0.2.2:8000
```

(`10.0.2.2` is the Android emulator's alias for the host machine's `localhost`.)

### Training Notebooks

The training and benchmarking notebooks in `notebooks/` are designed for
Google Colab with a free-tier T4 GPU. Open them in Colab, add a Hugging Face
token to Colab Secrets as `HF_TOKEN`, and run all cells. Each notebook is
self-contained — it installs dependencies, loads its dataset, trains or
evaluates the model, and produces a full metrics dashboard with loss curves,
confusion matrices, and a classification report.

---

## CI/CD

Four GitHub Actions workflows run on every push:

- **Backend CI** — flake8 linting and pytest test suite against Python 3.11
- **Frontend CI** — `flutter analyze` and `flutter test` against the pinned Flutter SDK
- **CodeQL** — weekly static security analysis on the Python codebase
- **Deploy to Render** — runs the full backend lint and test suite on `main` branch pushes, then triggers a Render deploy via webhook only if all checks pass

All workflows must pass before any code reaches the `main` branch. Over 600
CI runs were recorded across the project's six-month development period.

---

## Documentation

Full project documentation is in `docs/`:

- **Project Proposal** (October 2025) — initial scope and feasibility
- **Interim Report** (November 2025) — Phase 1 prototype write-up
- **Final Report** (April 2026) — complete dissertation covering literature
  review, system analysis, design, implementation, evaluation, and conclusions

---

## License and Acknowledgements

This is an academic final year project submitted in partial fulfilment of the
BSc in Computer Science at Technological University Dublin. The codebase is
not currently licensed for redistribution; please contact the author if you
would like to reuse any part of it.

Models trained by the project author are released on Hugging Face under
permissive terms. Datasets used (Baly et al. Article-Bias-Prediction, BABE,
FinancialPhraseBank) are property of their respective authors and are cited
in the dissertation references.

Special thanks to:

- **Eoin Rogers**, project supervisor at TU Dublin
- The maintainers of FastAPI, Flutter, Supabase, and Hugging Face for
  providing the open-source tooling on which NewsScope is built

---

## Contact

- **Author:** Christopher Noblett
- **Email:** [christophernoblett47@gmail.com](mailto:christophernoblett47@gmail.com)
- **GitHub:** [@C22454222](https://github.com/C22454222)
- **Institution:** Technological University Dublin
