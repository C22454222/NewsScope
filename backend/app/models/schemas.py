from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional
from uuid import UUID


# Article schemas used by the API and data layer
class ArticleBase(BaseModel):
    """
    Shared fields for article records.

    This base model is reused for create and read operations
    to keep the schema consistent across the codebase.
    """
    source: Optional[str] = None
    url: str
    bias_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    published_at: Optional[datetime] = None
    content: Optional[str] = None


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
    created_at: datetime

    # Allow construction from ORM/DB objects where attributes are present
    model_config = ConfigDict(from_attributes=True)


# User schemas for profile and preference management
class UserBase(BaseModel):
    """
    Base user fields shared by create and read operations.
    """
    email: str
    preferences: Optional[dict] = {}
    bias_profile: Optional[dict] = {}


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


# Source schemas representing news outlets (BBC, CNN, etc.)
class SourceBase(BaseModel):
    """
    Base fields for a news source record.
    """
    name: str
    country: Optional[str]
    bias_rating: Optional[str]


class Source(SourceBase):
    """
    Full source model as persisted in the database.
    """
    id: UUID

    model_config = ConfigDict(from_attributes=True)
