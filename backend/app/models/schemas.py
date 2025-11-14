from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from uuid import UUID


# Articles
class ArticleBase(BaseModel):
    source: Optional[str] = None
    url: str
    bias_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    published_at: Optional[datetime] = None


class ArticleCreate(ArticleBase):
    pass


class Article(ArticleBase):
    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# Users
class UserBase(BaseModel):
    email: str
    preferences: Optional[dict] = {}
    bias_profile: Optional[dict] = {}


class UserCreate(UserBase):
    id: UUID  # Firebase UID


class User(UserBase):
    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# Sources
class SourceBase(BaseModel):
    name: str
    country: Optional[str]
    bias_rating: Optional[str]


class Source(SourceBase):
    id: UUID

    class Config:
        from_attributes = True
