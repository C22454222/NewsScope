# app/routes/users.py
from fastapi import APIRouter
from app.db.supabase import supabase
from app.models.schemas import UserCreate

# Router for user profile and preference management
router = APIRouter()


@router.post("/users")
def add_user(user: UserCreate):
    """
    Create a new user record in Supabase.

    Called after successful Firebase authentication to
    store additional profile data on the backend.
    """
    response = supabase.table("users").insert(user.model_dump()).execute()
    return response.data


@router.get("/users/{uid}")
def get_user(uid: str):
    """
    Fetch a single user by ID.
    """
    response = (
        supabase.table("users")
        .select("*")
        .eq("id", uid)
        .execute()
    )
    return response.data


@router.put("/users/{uid}/preferences")
def update_preferences(uid: str, prefs: dict):
    """
    Update the user's saved preferences (e.g. region, topics).

    prefs is expected to be a JSON-serializable dictionary.
    """
    response = (
        supabase.table("users")
        .update({"preferences": prefs})
        .eq("id", uid)
        .execute()
    )
    return response.data


@router.put("/users/{uid}/bias_profile")
def update_bias_profile(uid: str, profile: dict):
    """
    Update the user's computed bias profile.

    The profile object stores aggregated reading patterns that
    will be visualised in the mobile app.
    """
    response = (
        supabase.table("users")
        .update({"bias_profile": profile})
        .eq("id", uid)
        .execute()
    )
    return response.data
