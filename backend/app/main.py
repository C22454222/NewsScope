from fastapi import FastAPI
from backend.app.routes import articles


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


# Health check route (used by Render)
@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(articles.router)
