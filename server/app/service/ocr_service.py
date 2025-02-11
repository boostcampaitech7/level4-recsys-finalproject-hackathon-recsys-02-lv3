import httpx
import re
from fastapi import HTTPException
from app.config.settings import Settings

class OCRService():
    def __init__(self):
        self.setting = Settings()

    def _clean_line(self, line):
        line = re.sub(r'\d+\s*▶*\s*동영상*', '', line)
        line = re.sub(r'[^\w\s&(),.]', '', line)
        line = re.sub(r'\d+\s*$', '', line)
        line = re.sub(r'(\d+\s*|E\s*)$', '', line)
        return line.strip()

    def _extract_tracks_and_artists(self, data):
        # 입력 데이터를 라인별로 나누고, 공백 라인은 제거
        lines = [line.strip() for line in data.strip().split("\n") if line.strip()]

        # 트랙명과 아티스트명 추출 (트랙명과 아티스트명이 번갈아가며 나옴)
        result = []
        for i in range(0, len(lines), 2):
            track = self._clean_line(lines[i])  # 트랙명
            artist = self._clean_line(lines[i+1]) if i+1 < len(lines) else ''  # 아티스트명
            result.append((track, artist))

        return result
    
    def replace_text(self, text: str):
        return re.sub(r'\.+$', '%', text).strip()
    
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
                        track_artist = self._extract_tracks_and_artists(result_text)
                        return track_artist
                    else:
                        raise HTTPException(status_code=response.status_code, detail=response.json())
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")