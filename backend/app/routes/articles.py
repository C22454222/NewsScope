# app/routes/articles.py
from fastapi import APIRouter
from app.db.supabase import supabase
from app.models.schemas import ArticleCreate

# Router for article-related endpoints
router = APIRouter()


@router.get("")
def get_articles():
    """
    Retrieve all articles from the database.

    This is used by the mobile app to populate the home feed.
    """
    response = supabase.table("articles").select("*").execute()
    return response.data


@router.post("")
def add_article(article: ArticleCreate):
    """
    Insert a single article record into the database.

    Primarily useful for testing or manual insertion rather than
    the main ingestion pipeline.
    """
    insert_response = (
        supabase.table("articles")
        .insert(
            {
                "source": article.source,
                "url": article.url,
                "bias_score": article.bias_score,
                "sentiment_score": article.sentiment_score,
                "published_at": article.published_at,
                "content": article.content,
            }
        )
        .execute()
    )
    return insert_response.data
