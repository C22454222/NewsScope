"""Database schema and constraint integration tests.

These tests run against a Supabase test project using the
SUPABASE_TEST_URL and SUPABASE_TEST_KEY environment variables.
Skip if not configured to keep CI green for contributors without
test-DB access.
"""
import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_TEST_URL"),
    reason="Supabase test credentials not configured",
)


@pytest.fixture
def test_db():
    from supabase import create_client
    return create_client(
        os.environ["SUPABASE_TEST_URL"],
        os.environ["SUPABASE_TEST_KEY"],
    )


def test_articles_url_unique_constraint(test_db):
    """Inserting two articles with the same URL must fail."""
    url = f"https://example.com/{uuid.uuid4()}"
    row = {"url": url, "title": "T1"}
    test_db.table("articles").insert(row).execute()
    with pytest.raises(Exception):
        test_db.table("articles").insert(row).execute()


def test_sources_bias_rating_check_constraint(test_db):
    """Sources table must reject ratings outside the five-point scale."""
    bad = {"name": f"test_{uuid.uuid4()}", "bias_rating": "Far-Left"}
    with pytest.raises(Exception):
        test_db.table("sources").insert(bad).execute()


def test_sources_valid_rating_accepted(test_db):
    name = f"test_{uuid.uuid4()}"
    good = {"name": name, "bias_rating": "Center"}
    result = test_db.table("sources").insert(good).execute()
    assert result.data
    test_db.table("sources").delete().eq("name", name).execute()


def test_credibility_score_default(test_db):
    """A new article without explicit credibility_score should default to 80."""
    url = f"https://example.com/{uuid.uuid4()}"
    inserted = test_db.table("articles").insert({"url": url, "title": "T"}).execute()
    aid = inserted.data[0]["id"]
    fetched = test_db.table("articles").select("*").eq("id", aid).single().execute()
    assert fetched.data["credibility_score"] == 80.0
    test_db.table("articles").delete().eq("id", aid).execute()


def test_reading_history_set_null_on_archive(test_db):
    """Deleting an article should set reading_history.article_id to NULL,
    not cascade-delete the history row."""
    url = f"https://example.com/{uuid.uuid4()}"
    art = test_db.table("articles").insert({"url": url, "title": "T"}).execute()
    aid = art.data[0]["id"]
    user_email = f"test_{uuid.uuid4()}@x.com"
    test_db.table("users").insert({"id": "test_uid", "email": user_email}).execute()
    rh = test_db.table("reading_history").insert({
        "user_id": "test_uid",
        "article_id": aid,
        "bias_score": 0.5,
        "source": "BBC",
    }).execute()
    rh_id = rh.data[0]["id"]
    test_db.table("articles").delete().eq("id", aid).execute()
    after = test_db.table("reading_history").select("*").eq("id", rh_id).single().execute()
    assert after.data["article_id"] is None
    assert after.data["bias_score"] == 0.5
    test_db.table("reading_history").delete().eq("id", rh_id).execute()
    test_db.table("users").delete().eq("email", user_email).execute()
