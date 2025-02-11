import httpx
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.orm import Session
from app.config.settings import Settings
from app.dto.auth import SpotifyTokenRequest
from app.service.spotify_service import SpotifyService
from app.service.user_service import UserService
from app.utils.utils import create_redirect_url
from db.database import SessionLocal
import logging

router = APIRouter()
setting = Settings()
spotify_service = SpotifyService()
logger = logging.getLogger("uvicorn")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/login")
async def login(dev: bool = Query(False)):
    frontend_url = setting.FRONT_DEV_URL if dev else setting.FRONT_BASE_URL
    auth_url = (
        f"{setting.SPOTIFY_AUTHENTICATION_URL}/authorize?"
        f"client_id={setting.CLIENT_ID}&"
        f"response_type=code&"
        f"redirect_uri={setting.REDIRECT_URI}&"
        f"scope=playlist-read-private playlist-modify-public playlist-modify-private "
        f"user-library-read user-library-modify user-read-private"
        f"&state={frontend_url}"
    )
    return RedirectResponse(url=auth_url)

@router.get("/callback")
async def get_user(code: str = Query(..., description="Authorization code from Spotify"), \
                   state: str = Query(...), db: Session = Depends(get_db)):
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
        user_images = user_info.get("images", [])
        user_img_url = user_images[0].get("url") if user_images else None

        user = UserService.get_or_create_user(db, spotify_id, access_token, refresh_token)
        embedding = UserService.user_has_embedding(user.user_id)

        frontend_url = create_redirect_url(user, state, embedding, user_img_url)
        return RedirectResponse(frontend_url)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        raise HTTPException(status_code=e.response.status_code, detail="Spotify API request failed")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@router.get("/user/{user_id}/embedding")
async def check_user_embedding(user_id: int):
    if UserService.user_has_embedding(user_id):
        return JSONResponse(status_code=200, content={"exist":True})
    else:
        return JSONResponse(status_code=200, content={"exist":False})