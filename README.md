# NewsScope: A Global, Bias-Aware News Aggregator & Analysis Application

[![Backend CI](https://github.com/C22454222/NewsScope/actions/workflows/backend-ci.yml/badge.svg)](https://github.com/C22454222/NewsScope/actions/workflows/backend-ci.yml)
[![Frontend CI](https://github.com/C22454222/NewsScope/actions/workflows/frontend-ci.yml/badge.svg)](https://github.com/C22454222/NewsScope/actions/workflows/frontend-ci.yml)

## Overview

NewsScope is a cross-platform news application designed to combat misinformation by collecting and analyzing news stories from diverse English-language sources worldwide. It leverages an automated AI/NLP pipeline to detect political bias and sentiment, presenting users with a transparent, analyzed view of global events.

**Live Backend API (Swagger):** [https://newsscope-backend.onrender.com/docs](https://newsscope-backend.onrender.com/docs)

## Key Features

* **Automated Ingestion:** Aggregates news every hour from NewsAPI, RSS feeds (BBC, RTÉ, GB News), and GDELT.
* **AI Analysis:** Uses Hugging Face Inference APIs (DistilBERT, PoliticalBiasBERT) to score articles on Sentiment (Positive/Negative) and Political Bias (Left/Right).
* **Flutter Mobile App:** Interactive frontend with dynamic badges ("Left-Leaning", "Positive") and date-grouped news feeds.
* **User Accounts:** Secure authentication via Firebase (Email & Google OAuth).
* **Archive System:** Automatically moves articles older than 30 days to Supabase Storage to maintain performance.

## Tech Stack

* **Frontend:** Flutter (Dart), Firebase Auth, Provider
* **Backend:** FastAPI (Python), APScheduler, Pydantic
* **Database:** Supabase (PostgreSQL + Storage Buckets)
* **AI/NLP Engine:** Hugging Face Inference API
* **Infrastructure:** Render (Hosting), Cloudflare, GitHub Actions (CI/CD)

## Repository Structure

* `backend/` - FastAPI server, ingestion scripts (`jobs/ingestion.py`), and analysis logic (`jobs/analysis.py`).
* `frontend/` - Flutter mobile application source code.
* `docs/` - Project documentation including Proposal, Feasibility Study, and Interim Report.
* `.github/workflows/` - Automated testing pipelines for CI/CD.

## Local Setup Instructions

### Prerequisites

* Flutter SDK installed (`flutter doctor` to verify)
* Python 3.10+ installed
* Git installed

### 1. Backend Setup

Navigate to the backend folder and install dependencies:

cd backend
python -m venv venv
source venv/bin/activate # (On Windows: venv\Scripts\activate)
pip install -r requirements.txt

Create a `.env` file in `backend/` with the API keys provided in the submission report.
Run the server locally:

Create a `.env` file in `backend/` with the API keys provided in the submission report.
Run the server locally:

uvicorn app.main:app --reload

### 2. Frontend Setup

Navigate to the frontend folder and run the app:

cd frontend/newsscope
flutter pub get
flutter run

## Authors

* **Christopher Noblett** (C22454222) - *Final Year Project 2025*
