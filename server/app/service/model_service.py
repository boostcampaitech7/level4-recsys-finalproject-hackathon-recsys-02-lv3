import httpx
from app.config.settings import Settings

class ModelService():
    def __init__(self):
        self.setting = Settings()

    async def make_request(self, method: str, url: str, **kwargs) -> dict:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(method, f"{self.setting.MODEL_API_URL}{url}", **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError:
                raise
