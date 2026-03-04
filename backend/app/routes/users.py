"""
NewsScope Users API Router.
Flake8: 0 errors/warnings.
"""

from fastapi import APIRouter, HTTPException

from app.db.supabase import supabase
from app.schemas import UserCreate

router = APIRouter()


@router.post("/users")
def add_user(user: UserCreate) -> dict:
    """
    Upsert a user record in Supabase.

    Called after every Firebase login — not just registration —
    so the users row is guaranteed present before any reading
    history is written. upsert on conflict id is idempotent.
    """
    response = (
        supabase.table("users")
        .upsert(user.model_dump(), on_conflict="id")
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=500, detail="User creation failed")
    return response.data[0]


@router.get("/users/{uid}")
def get_user(uid: str) -> dict:
    """Fetch a single user record by Firebase UID."""
    response = (
        supabase.table("users")
        .select("*")
        .eq("id", uid)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="User not found")
    return response.data[0]


@router.put("/users/{uid}/preferences")
def update_preferences(uid: str, prefs: dict) -> dict:
    """Update the user's saved preferences (e.g. region, topics)."""
    response = (
        supabase.table("users")
        .update({"preferences": prefs})
        .eq("id", uid)
        .execute()
    )
    if not response.data:
        raise HTTPException(
            status_code=404, detail="User not found or update failed"
        )
    return response.data[0]


@router.put("/users/{uid}/bias_profile")
def update_bias_profile(uid: str, profile: dict) -> dict:
    """Update the user's computed bias profile."""
    response = (
        supabase.table("users")
        .update({"bias_profile": profile})
        .eq("id", uid)
        .execute()
    )
    if not response.data:
        raise HTTPException(
            status_code=404, detail="User not found or update failed"
        )
    return response.data[0]
