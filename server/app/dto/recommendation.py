from pydantic import BaseModel
from typing import Optional
from app.dto.common import Artist

class PlaylistRecommendation(BaseModel):
    user_id: int
    items: list[int]

class OCRTrack(BaseModel):
    track_name: str
    artist_name: str

class OCRRecommendation(BaseModel):
    user_id: int
    items: list[OCRTrack]