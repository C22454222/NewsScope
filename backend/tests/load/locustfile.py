from locust import HttpUser, task, between, events


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Warm up the Render free-tier instance before the swarm."""
    print("Warming up the backend...")
    import time
    import requests
    try:
        requests.get(f"{environment.host}/articles?limit=1", timeout=30)
        print("Warmup complete.")
    except Exception as e:
        print(f"Warmup failed: {e}")
    time.sleep(10)


class NewsScopeUser(HttpUser):
    """Simulates a user opening the app and reading the home feed."""

    wait_time = between(5, 10)

    @task
    def open_app(self):
        with self.client.get(
            "/articles?limit=20",
            name="GET /articles",
            catch_response=True,
            timeout=15,
        ) as r:
            if r.status_code == 200:
                r.success()
            else:
                r.failure(f"Status {r.status_code}")
