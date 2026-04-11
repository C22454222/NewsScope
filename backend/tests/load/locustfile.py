"""Locust load test for the NewsScope backend."""
from locust import HttpUser, between, task


class NewsScopeUser(HttpUser):
    """Simulated user issuing requests against the public API."""

    wait_time = between(1, 3)

    @task(3)
    def browse_feed(self):
        self.client.get("/articles?limit=20")

    @task(2)
    def filter_by_category(self):
        self.client.get("/articles?category=politics&limit=20")

    @task(1)
    def compare_stories(self):
        self.client.post("/api/articles/compare", json={"topic": "climate"})

    @task(1)
    def get_sources(self):
        self.client.get("/sources")
