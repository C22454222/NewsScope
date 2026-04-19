"""
NewsScope sources API router.

Provides endpoints for listing and adding news source records.
"""

from fastapi import APIRouter

from app.db.supabase import supabase
from app.schemas import SourceBase

router = APIRouter()


@router.get("")
def get_sources() -> list:
    """
    Return all configured news sources.

    Used to power source filter UIs and source-level diagnostics
    in the Flutter client.
    """
    return supabase.table("sources").select("*").execute().data


@router.post("")
def add_source(source: SourceBase) -> list:
    """
    Add a new news source record.

    Converts the Pydantic model to a dict via model_dump() before
    inserting so only declared schema fields are written.
    """
    return (
        supabase.table("sources")
        .insert(source.model_dump())
        .execute()
        .data
    )
