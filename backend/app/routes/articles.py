from fastapi import APIRouter
from app.db.supabase import supabase
from app.models.schemas import ArticleCreate


router = APIRouter()


@router.get("/articles")
def get_articles():
    response = supabase.table("articles").select("*").execute()
    return response.data


@router.post("/articles")
def add_article(article: ArticleCreate):
    insert_response = supabase.table("articles").insert({
        "source": article.source,
        "url": article.url,
        "bias_score": article.bias_score,
        "sentiment_score": article.sentiment_score,
        "published_at": article.published_at
    }).execute()
    return insert_response.data
