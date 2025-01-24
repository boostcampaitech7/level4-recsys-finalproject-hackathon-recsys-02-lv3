import httpx
import base64
from app.config.settings import Settings
from app.dto.auth import SpotifyTokenRequest

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