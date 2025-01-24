import httpx
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from app.config.settings import Settings
from app.dto.auth import SpotifyTokenRequest
from app.utils.spotify_api_service import SpotifyApiService
from app.utils.user_service import UserService
from app.utils.utils import create_redirect_url
from db.database_postgres import PostgresSessionLocal
from db.database_embedding import EmbeddingSessionLocal
import logging

router = APIRouter()
setting = Settings()
spotify_service = SpotifyApiService()
logger = logging.getLogger("uvicorn")

def get_postgres_db():
    db = PostgresSessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_embedding_db():
    db = EmbeddingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/login")
async def login():
    auth_url = (
        f"{setting.SPOTIFY_AUTHENTICATION_URL}/authorize?"
        f"client_id={setting.CLIENT_ID}&"
        f"response_type=code&"
        f"redirect_uri={setting.REDIRECT_URI}&"
        f"scope=playlist-read-private playlist-modify-public playlist-modify-private "
        f"user-library-read user-library-modify user-read-private"
    )
    return RedirectResponse(url=auth_url)

@router.get("/callback")
async def get_user(code: str = Query(..., description="Authorization code from Spotify"), \
                        postgres_db: Session = Depends(get_postgres_db), 
                        embedding_db: Session = Depends(get_embedding_db)):
    token_request = SpotifyTokenRequest(
        code=code,
        redirect_uri=setting.REDIRECT_URI
    )

    try:
        tokens = await spotify_service.get_tokens(token_request)
        access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]

        user_info = await spotify_service.get_user_info(access_token)
        spotify_id = user_info["id"]
        user_img_url = user_info.get("images", [{}])[0].get("url")

        user = UserService.get_or_create_user(postgres_db, spotify_id, access_token, refresh_token)
        embedding = UserService.user_has_embedding(embedding_db, user.user_id)

        frontend_url = create_redirect_url(user, embedding, user_img_url)
        return RedirectResponse(frontend_url)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        raise HTTPException(status_code=e.response.status_code, detail="Spotify API request failed")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")