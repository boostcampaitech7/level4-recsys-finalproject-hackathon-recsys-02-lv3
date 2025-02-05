from pydantic import BaseModel
from typing import Optional

class Artist(BaseModel):
    artist_name: Optional[str]

class Recommendation(BaseModel):
    track_id: int
    track_name: str
    track_img_url: Optional[str] = None
    artists: list[Artist]
    description: str = "추천 이유 설명"

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

class InsertTrack(BaseModel):
    track_name: str
    artists: list[Artist]

class InsertTrackRequest(BaseModel):
    items: list[InsertTrack]