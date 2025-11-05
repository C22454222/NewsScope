from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from uuid import UUID


class ArticleBase(BaseModel):
    source: Optional[str] = None
    url: str
    bias_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    published_at: Optional[datetime] = None


class ArticleCreate(ArticleBase):
    pass  # same fields as base, used for POST


class Article(ArticleBase):
    id: UUID
    created_at: datetime

    class Config:
        orm_mode = True
        