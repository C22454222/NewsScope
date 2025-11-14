from fastapi import FastAPI
from app.routes import articles, users, sources


app = FastAPI(title="NewsScope API")


# Root route
@app.get("/")
def root():
    return {
        "message": "Welcome to NewsScope API. Try /health to check status."
    }


# HEAD route (for Render probes)
@app.head("/")
def root_head():
    return {}


# Health check route
@app.get("/health")
def health():
    return {"status": "ok"}


# Routers
app.include_router(articles.router)
app.include_router(users.router)
app.include_router(sources.router)