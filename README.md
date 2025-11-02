# NewsScope: A Global, Bias-Aware News Aggregator & Analysis Application

## Overview
NewsScope is a cross-platform news application designed to collect and analyse news stories from English-language sources worldwide.  
It uses AI/NLP to detect tone, political bias, and misinformation, presenting results in an interactive, user-friendly way.

## Tech Stack
- **Frontend:** Flutter (Dart), Firebase Auth
- **Backend:** FastAPI (Python), APScheduler
- **Database:** Supabase (Postgres + Buckets)
- **NLP Engine:** Hugging Face Inference API (RoBERTa, DistilBERT), TruSt, spaCy
- **Infra:** Render, Cloudflare, GitHub Actions

## Repository Structure
- `backend/` → FastAPI backend
- `frontend/` → Flutter app
- `docs/` → Diagrams, schema, and documentation
- `.github/workflows/` → CI/CD pipelines

## Getting Started
### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload