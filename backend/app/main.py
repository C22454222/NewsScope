from fastapi import FastAPI
# Test
app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "NewsScope backend is running"}
