# app/routes/sources.py
from fastapi import APIRouter
from app.db.supabase import supabase
from app.models.schemas import SourceBase

# Router for news source metadata (BBC, CNN, etc.)
router = APIRouter()


@router.get("/sources")
def get_sources():
    """
    Return the list of configured news sources.

    This can be used to power filter UIs or diagnostics.
    """
    response = supabase.table("sources").select("*").execute()
    return response.data


@router.post("/sources")
def add_source(source: SourceBase):
    """
    Add a new news source record.

    Uses Pydantic's model_dump() to convert the model to a dict.
    """
    response = supabase.table("sources").insert(source.model_dump()).execute()
    return response.data
