from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to NewsScope Backend"}

@app.get("/stories")
def get_stories():
    # Temporary mock data
    return [
        {"title": "Breaking News: Flutter + FastAPI!", "source": "Demo"},
        {"title": "Global Headlines Test", "source": "Mock API"},
        {"title": "Your backend is working!", "source": "FastAPI"},
    ]
