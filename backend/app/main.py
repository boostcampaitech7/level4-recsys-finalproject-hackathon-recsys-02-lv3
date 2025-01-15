from fastapi import FastAPI
from app.router import auth, playlist

app = FastAPI()

app.include_router(auth.router)
app.include_router(playlist.router)