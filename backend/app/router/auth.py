import httpx
import base64
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.orm import Session
from app.config.settings import Settings
from app.dto.auth import SpotifyTokenRequest, GetUserResponse
from db.database import SessionLocal, User

router = APIRouter()
setting = Settings()

def get_db():
    db = SessionLocal()
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
                        db: Session = Depends(get_db)):
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

            if user_info.status_code == 200:
                spotify_id = user_info.json()["id"]
                images = user_info.json().get("images", [])
                if images:
                    user_img_url = images[0]["url"]
                else:
                    user_img_url = None
            else:
                spotify_id = None
                user_img_url = None

            existing_user = db.query(User).filter(User.spotify_id == spotify_id).first()
            if existing_user:
                existing_user.access_token = access_token
                existing_user.refresh_token = refresh_token
                db.commit()
                db.refresh(existing_user)
                frontend_url = f"http://localhost:5173?user_id={existing_user.user_id}&user_img_url={user_img_url}"
                return RedirectResponse(frontend_url)
            else:
                # 새 사용자 추가
                new_user = User(spotify_id=spotify_id, access_token=access_token, refresh_token=refresh_token)
                db.add(new_user)
                db.commit()
                db.refresh(new_user)
                frontend_url = f"http://localhost:5173?user_id={new_user.user_id}&user_img_url={user_img_url}"
                return RedirectResponse(frontend_url)
        else:
            raise HTTPException(status_code=404, detail="failed to get access token from spotify")