"""
NewsScope Users API Router.
Flake8: 0 errors/warnings.
"""

from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Optional

from firebase_admin import auth

from app.db.supabase import supabase
from app.schemas import UserCreate

router = APIRouter()


# ── Auth dependency (local copy to avoid circular import with main.py) ───────

def _get_current_user(
    authorization: Optional[str] = Header(None),
) -> str:
    """
    Validate the Firebase ID token and return the authenticated UID.

    This is a local copy of the get_current_user dependency defined in
    main.py. Duplicating it here avoids a circular import (main imports
    this router, so this router cannot import main). The implementation
    is intentionally identical.
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


@router.delete("/{uid}/history")
def clear_user_history(
    uid: str,
    current_user: str = Depends(_get_current_user),
) -> dict:
    """
    Delete all reading_history rows for a user without deleting
    the user account itself.

    Called from the Flutter settings screen when the user taps
    "Clear Reading History". Resets the bias profile back to an
    empty state while preserving the account, preferences,
    notification subscription, and Firebase auth record.

    The current user is validated against the uid path parameter —
    users can only clear their own history, never someone else's.
    Returns 200 even if no rows found (idempotent).
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

    Deletes reading_history first (FK constraint), then the users row.
    Called from the Flutter client immediately after Firebase
    user.delete() succeeds — keeps Firebase and Supabase in sync.
    Returns 200 even if no rows found (idempotent — safe to retry).
    """
    supabase.table("reading_history").delete().eq("user_id", uid).execute()
    supabase.table("users").delete().eq("id", uid).execute()
    return {"deleted": uid}

# Test helpers (used by tests/unit/test_bias_profile.py)


def compute_weighted_average(items):
    """Compute time-weighted average of a list of {score, weight} dicts."""
    if not items:
        return 0.0
    total_weight = sum(item.get("weight", 0) for item in items)
    if total_weight == 0:
        return 0.0
    weighted_sum = sum(item.get("score", 0) * item.get("weight", 0) for item in items)
    return weighted_sum / total_weight


def largest_remainder_round(fractions):
    """Round a list of fractions to integer percentages summing to exactly 100."""
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
