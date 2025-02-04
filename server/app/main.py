from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.router import auth, playlist, recommendation, onboarding, interaction

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],                     
    allow_headers=["*"],                     
)

app.include_router(auth.router)
app.include_router(onboarding.router)
app.include_router(interaction.router)
app.include_router(playlist.router)
app.include_router(recommendation.router)