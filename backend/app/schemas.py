# app/schemas.py
"""
NewsScope Pydantic schemas.
Flake8: 0 errors.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ArticleBase(BaseModel):
    """
    Shared fields for article records.

    Reused for create and read operations to keep the schema
    consistent across the codebase.
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


class ArticleCreate(ArticleBase):
    """Payload expected when creating a new article via the API."""

    pass


class Article(ArticleBase):
    """Full article model as stored and returned by the backend."""

    id: UUID
    source_id: Optional[UUID] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ArticleResponse(ArticleBase):
    """
    Full article response model returned by the API and used
    internally by fact_checking.py and ingestion.py.

    Extends ArticleBase with DB-generated fields and Week 4
    credibility/fact-check columns. extra='ignore' silently drops
    unknown Supabase columns so ArticleResponse(**row) is always safe.
    """

    id: Optional[str] = None
    source_id: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[str] = None

    # Week 4 — credibility + fact-checking
    credibility_score: Optional[float] = 80.0
    fact_checks: Optional[Dict[str, Any]] = Field(default_factory=dict)
    claims_checked: Optional[int] = 0
    credibility_reason: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class UserBase(BaseModel):
    """Base user fields shared by create and read operations."""

    email: str
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
    bias_profile: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UserCreate(UserBase):
    """
    Payload used when creating a new user record.

    The id field is the Firebase UID used as the primary key
    in Supabase.
    """

    id: UUID


class User(UserBase):
    """Full user model as returned by the backend."""

    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SourceBase(BaseModel):
    """Base fields for a news source record."""

    name: str
    country: Optional[str] = None
    bias_rating: Optional[str] = None


class Source(SourceBase):
    """Full source model as persisted in the database."""

    id: UUID

    model_config = ConfigDict(from_attributes=True)


class ReadingHistoryBase(BaseModel):
    """Base fields for reading history tracking."""

    article_id: UUID
    time_spent_seconds: int


class ReadingHistoryCreate(BaseModel):
    """
    Payload for tracking article reading time.
    Sent from the mobile app when the user exits an article.
    """

    article_id: str
    time_spent_seconds: int


class ReadingHistory(ReadingHistoryBase):
    """Full reading history record as stored in the database."""

    id: UUID
    user_id: UUID
    opened_at: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BiasProfile(BaseModel):
    """
    Calculated bias profile based on the user's reading history,
    weighted by time spent on articles.
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


class FactCheckBase(BaseModel):
    """Base fields for fact-check records."""

    claim: str
    rating: Optional[str] = None
    source: Optional[str] = None
    link: Optional[str] = None
    politifact_url: Optional[str] = None


class FactCheckCreate(FactCheckBase):
    """Payload for creating a fact-check record."""

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
    politifact_url: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ComparisonRequest(BaseModel):
    """Request payload for comparing coverage across outlets."""

    topic: str
    limit: int = Field(default=5, ge=1, le=20)


class ComparisonResponse(BaseModel):
    """Response containing articles grouped by political leaning."""

    topic: str
    left_articles: List[Article]
    center_articles: List[Article]
    right_articles: List[Article]
    total_found: int


class UserFactCheckView(BaseModel):
    """Track when users view fact-checks."""

    id: UUID
    user_id: UUID
    fact_check_id: UUID
    viewed_at: datetime

    model_config = ConfigDict(from_attributes=True)
