import httpx
import re
from fastapi import HTTPException
from app.config.settings import Settings

class OCRService():
    def __init__(self):
        self.setting = Settings()

    def _text_preprocessing(self, text: str):
        text = re.sub(r'동영상\s*·?\s*', '', text)
        text = re.sub(r'^19\s*', '', text).strip()
        pattern = r'([^\n·]+)\s*·?\s*([^\n·]+)'
        matches = re.findall(pattern, text)
        return matches

    async def make_request(self, image) -> dict:
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                files = {"document": (image.filename, await image.read(), image.content_type)}
                headers = {
                    "Authorization": f"Bearer {self.setting.UPSTAGE_API_KEY}"
                }
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.setting.UPSTAGE_OCR_API_URL}",
                        headers=headers,
                        files=files
                    )
                    if response.status_code == 200:
                        result_text = response.json()["text"]
                        track_artist = self._text_preprocessing(result_text)
                        return track_artist
                    else:
                        raise HTTPException(status_code=response.status_code, detail=response.json())
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")