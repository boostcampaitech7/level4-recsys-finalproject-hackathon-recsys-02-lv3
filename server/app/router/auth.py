import httpx
import base64
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from app.config.settings import Settings
from app.dto.auth import SpotifyTokenRequest
from db.database_postgres import PostgresSessionLocal, User
from db.database_embedding import EmbeddingSessionLocal, User_Emb
import logging

router = APIRouter()
setting = Settings()
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
    accesstoken_request = SpotifyTokenRequest(
        code=code,
        redirect_uri=setting.REDIRECT_URI
    )

    authorization = f"{setting.CLIENT_ID}:{setting.CLIENT_SECRET}"
    authorization_encoding = base64.b64encode(authorization.encode()).decode('utf-8')

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{setting.SPOTIFY_AUTHENTICATION_URL}/api/token",
            headers={
                "Authorization":f"Basic {authorization_encoding}",
                "Content-type": "application/x-www-form-urlencoded"
            },
            data=accesstoken_request.dict()
        )

        if response.status_code == 200:
            tokens = response.json()
            access_token = tokens["access_token"]
            refresh_token = tokens["refresh_token"]

            user_info = await client.get(
                f"{setting.SPOTIFY_API_URL}/me",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            logger.info(f"Response content: {response.text}")
            logger.info(f"Response content: {user_info.text}")

            if user_info.status_code == 200:
                spotify_id = user_info.json()["id"]
                images = user_info.json().get("images", [])
                if images:
                    user_img_url = images[0]["url"]
                else:
                    user_img_url = None
                user = postgres_db.query(User).filter(User.spotify_id == spotify_id).first()
                if user:
                    # 기존 사용자 업데이트 
                    user.access_token = access_token
                    user.refresh_token = refresh_token
                    postgres_db.commit()
                    postgres_db.refresh(user)
                else:
                    # 새 사용자 추가
                    user = User(spotify_id=spotify_id, access_token=access_token, refresh_token=refresh_token)
                    postgres_db.add(user)
                    postgres_db.commit()
                    postgres_db.refresh(user)
                user_embedding = embedding_db.query(User_Emb).filter(User_Emb.user_id == user.user_id).first()
                if user_embedding:
                    frontend_url = f"http://localhost:5173/home?user_id={user.user_id}"
                else:
                    frontend_url = f"http://localhost:5173/onboarding?user_id={user.user_id}"
                if user_img_url:
                    frontend_url += f"&user_img_url={user_img_url}"
                return RedirectResponse(frontend_url)
            else:
                raise HTTPException(status_code=401, detail="cannot access to spotify")
        else:
            raise HTTPException(status_code=404, detail="failed to get access token from spotify")