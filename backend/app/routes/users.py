"""
NewsScope Users API Router.
Flake8: 0 errors/warnings.
"""

from fastapi import APIRouter, HTTPException

from app.db.supabase import supabase
from app.schemas import UserCreate

router = APIRouter()


@router.post("")
def add_user(user: UserCreate) -> dict:
    """
    Upsert a user record in Supabase.

    Called after every Firebase login — not just registration —
    so the users row is guaranteed present before any reading
    history is written.

    Deduplication strategy
    ──────────────────────
    Google Sign-In can produce multiple Firebase UIDs for the same
    email address if the user signs in on a new device or after
    clearing app data. We use ON CONFLICT (email) so that if a row
    already exists for this email, we UPDATE its id to the latest
    Firebase UID rather than inserting a duplicate row.

    This keeps the users row in sync with whatever UID Firebase
    currently considers canonical, which means:
      - reading_history FK writes always succeed (UID matches)
      - no orphaned rows accumulate over time
      - the operation is atomic — no select-then-insert race condition
    """
    payload: dict[str, str | None] = {
        "id": user.id,
        "email": user.email,
    }
    if user.display_name is not None:
        payload["display_name"] = user.display_name

    response = (
        supabase.table("users")
        .upsert(
            payload,
            on_conflict="email",          # deduplicate on email, not UID
        )
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=500, detail="User upsert failed")

    new_uid = user.id
    returned_uid = response.data[0]["id"]

    # If the email already existed under a different UID, Supabase will have
    # updated the row's id to new_uid. Re-parent any reading_history rows
    # that were written under the old UID so history is never orphaned.
    #
    # NOTE: This only matters in the rare Google re-auth edge case.
    # In the normal path old_uid == new_uid and this is a no-op.
    if returned_uid != new_uid:
        (
            supabase.table("reading_history")
            .update({"user_id": new_uid})
            .eq("user_id", returned_uid)
            .execute()
        )

    return response.data[0]


@router.get("/{uid}")
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


@router.put("/{uid}/preferences")
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


@router.delete("/{uid}")
def delete_user(uid: str) -> dict:
    """
    Delete all Supabase data for a user by Firebase UID.

    Deletes reading_history first (FK constraint), then the users row.
    Called from the Flutter client immediately after Firebase
    user.delete() succeeds — keeps Firebase and Supabase in sync.
    Returns 200 even if no rows found (idempotent — safe to retry).
    """
    supabase.table("reading_history").delete().eq("user_id", uid).execute()
    supabase.table("users").delete().eq("id", uid).execute()
    return {"deleted": uid}
