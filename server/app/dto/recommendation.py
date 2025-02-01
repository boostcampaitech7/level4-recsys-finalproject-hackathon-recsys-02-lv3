from pydantic import BaseModel
from typing import Optional
from app.dto.playlist import Artist

class OnboardingRequest(BaseModel):
    user_id: int
    tags: list[str]

class PlaylistRecommendation(BaseModel):
    user_id: int
    items: list[int]

class TrackMetaData(BaseModel):
    track_name: str
    artist_name: str
    playlist_name: Optional[str] = "내 플레이리스트"
    genres: list[str]
    length: int
    listeners: int

class RecommendationRequest(BaseModel):
    user_id: int
    exists: list[int]
    missing: list[TrackMetaData]

class TrackIdPair(BaseModel):
    item1: int
    item2: int

class OCRTrack(BaseModel):
    track_name: str
    artist_name: str

class OCRRecommendation(BaseModel):
    user_id: int
    items: list[OCRTrack]

class Onboarding(BaseModel):
    track_id: int
    track_name: str
    track_img_url: Optional[str] = None
    artists: list[Artist]
    tags: list[str] = ["신나는"]

class Recommendation(BaseModel):
    track_id: int
    track_name: str
    track_img_url: Optional[str] = None
    artists: list[Artist]
    description: str = "추천 이유 설명"