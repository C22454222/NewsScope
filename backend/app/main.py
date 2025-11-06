from fastapi import FastAPI
from app.routes import articles


app = FastAPI()


# Root route (so hitting "/" doesn't 404)
@app.get("/")
def root():
    return {
        "message": (
            "Welcome to NewsScope API. "
            "Try /health to check status."
        )
    }

# Add HEAD route so Render probes don't trigger 405
@app.head("/")
def root_head():
    # HEAD responses should be empty, just return 200 OK
    return {}

# Health check route (used by Render)
@app.get("/health")
def health():
    return {"status": "ok"}

# Include your articles router
app.include_router(articles.router)
