"""
NewsScope Pydantic schemas.

Matches the Supabase PostgreSQL schema exactly:
  - articles: all NLP output columns including bias_explanation JSONB
  - fact_checks: relational table with article_id FK
  - reading_history: snapshot columns for bias profile calculation
    (bias_score, sentiment_score, source, general_bias, credibility_score)
  - sources: outlet metadata with bias_rating check constraint
  - users: id (Firebase UID), email, created_at, updated_at, display_name

No description column — not in the database.
No preferences/bias_profile on users — profile is calculated live
from reading_history; preferences are stored client-side via
SharedPreferences.

Flake8: 0 errors/warnings.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ── Articles ──────────────────────────────────────────────────────────────────


class ArticleBase(BaseModel):
    """
    Shared fields for article records.

    Mirrors the articles table columns exactly.
    Both bias_score (source-level, [-1,+1]) and political_bias /
    political_bias_score (article-level RoBERTa output) are present —
    they are distinct fields written by different parts of analysis.py.
    """

    source: Optional[str] = None
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    bias_score: Optional[float] = None
    bias_intensity: Optional[float] = None
    sentiment_score: Optional[float] = None
    published_at: Optional[datetime] = None
    category: Optional[str] = None
    general_bias: Optional[str] = None
    general_bias_score: Optional[float] = None
    political_bias: Optional[str] = None
    political_bias_score: Optional[float] = None


class ArticleCreate(ArticleBase):
    """Payload expected when creating a new article via the API."""

    pass


class Article(ArticleBase):
    """Full article model as stored and returned by the backend."""

    id: UUID
    source_id: Optional[UUID] = None
    created_at: datetime
    updated_at: Optional[datetime] = None          # DB: updated_at timestamptz

    model_config = ConfigDict(from_attributes=True)


class ArticleResponse(ArticleBase):
    """
    Full article response model returned by the API and used
    internally by fact_checking.py and ingestion.py.

    extra='ignore' silently drops unknown Supabase columns so
    ArticleResponse(**row) is always safe even as the schema evolves.

    credibility_score is seeded at 80.0 by ingestion and replaced
    by the real computed value once fact_checking.py runs.

    credibility_reason is a plain human-readable string, e.g.
    "2/3 queries matched fact-checks. Source: BBC News. Rated reliable."

    bias_explanation is a JSONB list of word-weight dicts produced
    by LIME in analysis.py. None until analysis scores the article
    with political_bias_score >= 0.6. Shape per item:
      {"word": str, "weight": float, "direction": "towards"|"against"}
    """

    id: Optional[str] = None
    source_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None          # fixed: was Optional[str]

    # Credibility + fact-checking fields
    credibility_score: Optional[float] = None
    fact_checks: Optional[Dict[str, Any]] = Field(default_factory=dict)
    claims_checked: Optional[int] = 0
    credibility_reason: Optional[str] = None
    credibility_updated_at: Optional[datetime] = None

    # LIME bias explainability
    bias_explanation: Optional[List[Dict[str, Any]]] = None

    model_config = ConfigDict(from_attributes=True, extra="ignore")


# ── Users ─────────────────────────────────────────────────────────────────────


class UserCreate(BaseModel):
    """
    Payload used when creating or upserting a user record.

    Matches the users table: id, email, display_name.
    updated_at is managed server-side by the set_updated_at trigger
    and is never sent by the client.
    """

    id: str                                        # Firebase UID — plain string
    email: str
    display_name: Optional[str] = None            # added: column exists in DB


class User(BaseModel):
    """Full user model as returned by the backend."""

    id: str                                        # Firebase UID
    email: str
    display_name: Optional[str] = None            # added: column exists in DB
    created_at: datetime
    updated_at: Optional[datetime] = None         # added: column exists in DB

    model_config = ConfigDict(from_attributes=True)


# ── Sources ───────────────────────────────────────────────────────────────────


class SourceBase(BaseModel):
    """Base fields for a news source record."""

    name: str
    country: Optional[str] = None
    bias_rating: Optional[str] = None


class Source(SourceBase):
    """Full source model as persisted in the database."""

    id: UUID

    model_config = ConfigDict(from_attributes=True)


# ── Reading history ───────────────────────────────────────────────────────────


class ReadingHistoryCreate(BaseModel):
    """
    Payload for tracking article reading time.
    Sent from the mobile app when the user exits an article.
    article_id is a string here because the Flutter client sends
    the UUID as a plain string in the JSON body.
    """

    article_id: str
    time_spent_seconds: int


class ReadingHistory(BaseModel):
    """Full reading history record as stored in the database."""

    id: UUID
    user_id: str                                   # Firebase UID — text in DB
    article_id: Optional[UUID] = None
    time_spent_seconds: int
    opened_at: datetime
    created_at: datetime
    bias_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    source: Optional[str] = None
    general_bias: Optional[str] = None
    credibility_score: Optional[float] = None     # added: snapshot column

    model_config = ConfigDict(from_attributes=True)


# ── Bias profile ──────────────────────────────────────────────────────────────


class BiasProfile(BaseModel):
    """
    Calculated bias profile returned by GET /api/bias-profile.

    Computed live from reading_history snapshot columns — never stored.
    source_breakdown: top-12 sources by article count.
    Powers the bar chart on the Flutter profile screen.

    avg_credibility: mean credibility score of articles the user has
    read (0-100 scale). Uses the snapshot column in reading_history so
    it stays stable after articles are archived. None when the user has
    no scored articles yet.
    """

    avg_bias: float
    avg_sentiment: float
    total_articles_read: int
    left_count: int
    center_count: int
    right_count: int
    most_read_source: str
    bias_distribution: Dict[str, float]
    reading_time_total_minutes: int
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    source_breakdown: Optional[Dict[str, int]] = None
    avg_credibility: Optional[float] = None       # added: powers profile header


# ── Fact checks ───────────────────────────────────────────────────────────────


class FactCheckBase(BaseModel):
    """Base fields for a fact-check record."""

    claim: str
    rating: Optional[str] = None
    source: Optional[str] = None
    link: Optional[str] = None


class FactCheckCreate(FactCheckBase):
    """Payload for inserting a fact-check row."""

    article_id: Optional[UUID] = None


class FactCheck(BaseModel):
    """Full fact-check record as stored in the database."""

    id: UUID
    article_id: Optional[UUID] = None
    checked_at: Optional[datetime] = None
    claim: str
    rating: Optional[str] = None
    source: Optional[str] = None
    link: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ── Comparison ────────────────────────────────────────────────────────────────


class ComparisonRequest(BaseModel):
    """
    Request payload for POST /api/articles/compare.

    topic searches title and content via ilike.
    category filters by the category column (sub-categories resolved
    via CATEGORY_GROUP_MAP in articles.py before the DB query).
    source filters by exact source name as stored in articles.source.

    limit is intentionally absent — the compare endpoint controls its
    own per-band limits internally and does not expose them to clients.
    """

    topic: str
    category: Optional[str] = None
    source: Optional[str] = None


class ComparisonResponse(BaseModel):
    """Articles grouped by political leaning for the compare screen."""

    topic: str
    left_articles: List[Dict[str, Any]]
    center_articles: List[Dict[str, Any]]
    right_articles: List[Dict[str, Any]]
    total_found: int
