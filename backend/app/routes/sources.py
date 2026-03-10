"""
NewsScope Sources API Router.
Flake8: 0 errors/warnings.
"""

from fastapi import APIRouter

from app.db.supabase import supabase
from app.schemas import SourceBase


router = APIRouter()


@router.get("")
def get_sources() -> list:
    """
    Return the list of configured news sources.

    Used to power filter UIs and source diagnostics.
    """
    return supabase.table("sources").select("*").execute().data


@router.post("")
def add_source(source: SourceBase) -> list:
    """
    Add a new news source record.

    Uses model_dump() to convert the Pydantic model to a dict.
    """
    return (
        supabase.table("sources")
        .insert(source.model_dump())
        .execute()
        .data
    )
