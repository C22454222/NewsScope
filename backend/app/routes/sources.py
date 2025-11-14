from fastapi import APIRouter
from app.db.supabase import supabase
from app.models.schemas import SourceBase


router = APIRouter()


@router.get("/sources")
def get_sources():
    response = supabase.table("sources").select("*").execute()
    return response.data


@router.post("/sources")
def add_source(source: SourceBase):
    # Use model_dump() instead of dict()
    response = supabase.table("sources").insert(source.model_dump()).execute()
    return response.data