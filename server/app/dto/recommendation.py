from pydantic import BaseModel
from typing import Optional
from app.dto.common import Artist

class PlaylistRecommendation(BaseModel):
    user_id: int
    items: list[int]

class TrackMetaData(BaseModel):
    track_name: str
    artists_name: str
    playlist_name: Optional[str] = " "
    genres: list[str]
    length: int
    listeners: int

class RecommendationRequest(BaseModel):
    user_id: int
    exists: list[int]
    missing: list[TrackMetaData]

class OCRTrack(BaseModel):
    track_name: str
    artist_name: str

class OCRRecommendation(BaseModel):
    user_id: int
    items: list[OCRTrack]

class Recommendation(BaseModel):
    track_id: int
    track_name: str
    track_img_url: Optional[str] = None
    artists: list[Artist]
    description: str = "추천 이유 설명"