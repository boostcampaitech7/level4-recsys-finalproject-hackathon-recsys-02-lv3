import httpx
import re
from fastapi import HTTPException
from app.config.settings import Settings

class OCRService():
    def __init__(self):
        self.setting = Settings()

    def _clean_name(self, name: str) -> str:
        """각 이름에서 '동영상'과 '19'를 제거"""
        name = re.sub(r'동영상\s*', '', name).strip()  # 맨 앞의 '동영상' 제거
        name = re.sub(r'^19\s*', '', name).strip() # 맨 앞의 '19' 제거거
        return name

    def _text_preprocessing(self, text: str):
        pattern = r'([^\n·]+)\s*·?\s*([^\n·]+)'
        matches = re.findall(pattern, text)
        # 추출된 결과를 정제하여 반환
        cleaned_matches = [(self._clean_name(track), self._clean_name(artist)) for track, artist in matches]
        return cleaned_matches
    
    def replace_text(self, text: str):
        return text.replace('...', '%').strip()
    
    def remove_text(self, text: str):
        return text.replace('%', '').strip()
    
    def normalize_text(self, text: str):
        text = text.lower().strip()
        text = text.replace(" ", "")  # 공백 제거
        return text

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