from fastapi import APIRouter
from app.db.supabase import supabase
from app.models.schemas import UserCreate


router = APIRouter()


@router.post("/users")
def add_user(user: UserCreate):
    response = supabase.table("users").insert(user.dict()).execute()
    return response.data


@router.get("/users/{uid}")
def get_user(uid: str):
    response = supabase.table("users").select("*").eq("id", uid).execute()
    return response.data


@router.put("/users/{uid}/preferences")
def update_preferences(uid: str, prefs: dict):
    response = supabase.table("users").update({"preferences": prefs}).eq("id", uid).execute()
    return response.data


@router.put("/users/{uid}/bias_profile")
def update_bias_profile(uid: str, profile: dict):
    response = supabase.table("users").update({"bias_profile": profile}).eq("id", uid).execute()
    return response.data
