from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID


class ArticleBase(BaseModel):
    """
    Shared fields for article records.

    This base model is reused for create and read operations
    to keep the schema consistent across the codebase.
    """
    source: Optional[str] = None
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    bias_score: Optional[float] = None
    bias_intensity: Optional[float] = None
    sentiment_score: Optional[float] = None
    published_at: Optional[datetime] = None
    category: Optional[str] = None  # ← ADDED


class ArticleCreate(ArticleBase):
    """
    Payload expected when creating a new article via the API.
    """
    pass


class Article(ArticleBase):
    """
    Full article model as stored and returned by the backend.
    """
    id: UUID
    source_id: Optional[UUID] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserBase(BaseModel):
    """
    Base user fields shared by create and read operations.
    """
    email: str
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
    bias_profile: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UserCreate(UserBase):
    """
    Payload used when creating a new user record.

    The id field is the Firebase UID, which is used as the
    primary key in Supabase.
    """
    id: UUID  # Firebase UID


class User(UserBase):
    """
    Full user model as returned by the backend.
    """
    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SourceBase(BaseModel):
    """
    Base fields for a news source record.
    """
    name: str
    country: Optional[str] = None
    bias_rating: Optional[str] = None


class Source(SourceBase):
    """
    Full source model as persisted in the database.
    """
    id: UUID

    model_config = ConfigDict(from_attributes=True)


class ReadingHistoryBase(BaseModel):
    """
    Base fields for reading history tracking.
    """
    article_id: UUID
    time_spent_seconds: int


class ReadingHistoryCreate(BaseModel):
    """
    Payload for tracking article reading time.
    Sent from mobile app when user exits article.
    """
    article_id: str
    time_spent_seconds: int


class ReadingHistory(ReadingHistoryBase):
    """
    Full reading history record as stored in database.
    """
    id: UUID
    user_id: UUID
    opened_at: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BiasProfile(BaseModel):
    """
    Calculated bias profile based on user's reading history.
    Weighted by time spent on articles.
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
    """
    Base fields for fact-check records.
    """
    claim: str
    rating: Optional[str] = None
    source: Optional[str] = None
    link: Optional[str] = None
    politifact_url: Optional[str] = None


class FactCheckCreate(FactCheckBase):
    """
    Payload for creating a fact-check record.
    """
    article_id: Optional[UUID] = None


class FactCheck(BaseModel):
    """
    Full fact-check record as stored in database.
    """
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
    """
    Request payload for comparing coverage across outlets.
    """
    topic: str
    limit: int = Field(default=5, ge=1, le=20)


class ComparisonResponse(BaseModel):
    """
    Response containing articles grouped by political leaning.
    """
    topic: str
    left_articles: List[Article]
    center_articles: List[Article]
    right_articles: List[Article]
    total_found: int


class UserFactCheckView(BaseModel):
    """
    Track when users view fact-checks.
    """
    id: UUID
    user_id: UUID
    fact_check_id: UUID
    viewed_at: datetime

    model_config = ConfigDict(from_attributes=True)
