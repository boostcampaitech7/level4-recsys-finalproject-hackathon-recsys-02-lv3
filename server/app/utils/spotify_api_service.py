import httpx
import base64
from app.config.settings import Settings
from app.dto.auth import SpotifyTokenRequest, SpotifyRefreshingTokenRequest

class SpotifyApiService:
    def __init__(self):
        self.setting = Settings()

    async def get_tokens(self, token_request: SpotifyTokenRequest) -> dict:
        authorization = f"{self.setting.CLIENT_ID}:{self.setting.CLIENT_SECRET}"
        authorization_encoding = base64.b64encode(authorization.encode()).decode('utf-8')

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.setting.SPOTIFY_AUTHENTICATION_URL}/api/token",
                headers={
                    "Authorization": f"Basic {authorization_encoding}",
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                data=token_request.dict()
            )
            response.raise_for_status()
            return response.json()
        
    async def get_user_info(self, access_token: str) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.setting.SPOTIFY_API_URL}/me",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            response.raise_for_status()
            return response.json()
        
    async def _refresh_access_token(self, user, db) -> str:
        authorization = f"{self.setting.CLIENT_ID}:{self.setting.CLIENT_SECRET}"
        authorization_encoding = base64.b64encode(authorization.encode()).decode('utf-8')

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.setting.SPOTIFY_AUTHENTICATION_URL}/api/token",
                headers={"Authorization": f"Basic {authorization_encoding}"},
                data=SpotifyRefreshingTokenRequest(
                    refresh_token=user.refresh_token
                ).dict(),
            )
            response.raise_for_status()
            tokens = response.json()
            user.access_token = tokens["access_token"]
            db.commit()
            db.refresh(user)
            return tokens["access_token"]

    async def _make_request(self, method: str, url: str, user, db, **kwargs) -> dict:
        async with httpx.AsyncClient() as client:
            headers = kwargs.pop("headers", {})
            headers["Authorization"] = f"Bearer {user.access_token}"

            try:
                response = await client.request(method, url, headers=headers, **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    user.access_token = await self._refresh_access_token(user, db)
                    headers["Authorization"] = f"Bearer {user.access_token}"
                    response = await client.request(method, url, headers=headers, **kwargs)
                    response.raise_for_status()
                    return response.json()
                raise