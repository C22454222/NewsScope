"""
NewsScope users API router.

Handles user creation, retrieval, deletion, and reading history
management. Firebase ID token validation is performed inline to
avoid a circular import with main.py.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from firebase_admin import auth

from app.db.supabase import supabase
from app.schemas import UserCreate

router = APIRouter()


def _get_current_user(
    authorization: Optional[str] = Header(None),
) -> str:
    """
    Validate the Firebase ID token and return the authenticated UID.

    This is a local copy of the dependency defined in main.py.
    Duplicating it here avoids a circular import -- main imports this
    router, so this router cannot import main. The implementation is
    intentionally identical.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.replace("Bearer ", "")
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token["uid"]
    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=401, detail="Invalid authentication token"
        )
    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=401, detail="Authentication token expired"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=401, detail=f"Authentication failed: {exc}"
        )


@router.post("")
def add_user(user: UserCreate) -> dict:
    """
    Upsert a user record in Supabase.

    Called after every Firebase login -- not just on first registration
    -- so the users row is guaranteed to exist before any reading
    history is written.

    Google Sign-In can produce multiple Firebase UIDs for the same email
    address across devices or after clearing app data. ON CONFLICT (email)
    updates the row id to the latest UID rather than inserting a duplicate,
    keeping reading_history FK writes safe and preventing orphaned rows.
    The upsert is atomic -- no select-then-insert race condition.

    If the returned UID differs from the new UID, any reading_history
    rows written under the old UID are re-parented to the new one so
    history is never lost. In the common path old and new UIDs are equal
    and this is a no-op.
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
            on_conflict="email",  # deduplicate on email, not UID
        )
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=500, detail="User upsert failed")

    new_uid = user.id
    returned_uid = response.data[0]["id"]

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


@router.delete("/{uid}/history")
def clear_user_history(
    uid: str,
    current_user: str = Depends(_get_current_user),
) -> dict:
    """
    Delete all reading_history rows for a user without removing the account.

    Called from the Flutter settings screen when the user taps
    "Clear Reading History". Resets the bias profile to empty while
    preserving the account, preferences, notification subscription,
    and Firebase auth record. Users can only clear their own history.
    Returns 200 even if no rows exist (idempotent).
    """
    if current_user != uid:
        raise HTTPException(
            status_code=403,
            detail="Cannot clear history for another user",
        )

    supabase.table("reading_history").delete().eq("user_id", uid).execute()
    return {"cleared": uid}


@router.delete("/{uid}")
def delete_user(uid: str) -> dict:
    """
    Delete all Supabase data for a user by Firebase UID.

    Deletes reading_history first to satisfy the FK constraint, then
    removes the users row. Called from the Flutter client immediately
    after Firebase user.delete() succeeds to keep both stores in sync.
    Returns 200 even if no rows exist (idempotent -- safe to retry).
    """
    supabase.table("reading_history").delete().eq("user_id", uid).execute()
    supabase.table("users").delete().eq("id", uid).execute()
    return {"deleted": uid}


def compute_weighted_average(items):
    """
    Compute the time-weighted average of a list of {score, weight} dicts.
    Used by tests/unit/test_bias_profile.py.
    """
    if not items:
        return 0.0
    total_weight = sum(item.get("weight", 0) for item in items)
    if total_weight == 0:
        return 0.0
    weighted_sum = sum(
        item.get("score", 0) * item.get("weight", 0) for item in items
    )
    return weighted_sum / total_weight


def largest_remainder_round(fractions):
    """
    Round a list of fractions to integer percentages that sum to exactly 100.
    Used by tests/unit/test_bias_profile.py.
    """
    if not fractions:
        return []
    scaled = [f * 100 for f in fractions]
    floored = [int(s) for s in scaled]
    remainder = 100 - sum(floored)
    remainders = sorted(
        [(scaled[i] - floored[i], i) for i in range(len(scaled))],
        reverse=True,
    )
    for _, i in remainders[:remainder]:
        floored[i] += 1
    return floored
