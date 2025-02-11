from pydantic import BaseModel, Field
from typing import List, Dict

class RecommendationRequest(BaseModel):
    user_id: int
    items: List[int]

class RecommendationResponse(BaseModel):
    items: List[int]

class OnboardingData(BaseModel):
    user_id: int = Field(..., description="사용자 ID")
    tags: List[int] = Field(..., description="사용자 선호 태그 목록")

class TrackPair(BaseModel):
    item1: int
    item2: int

class OnboardingResponse(BaseModel):
    items: List[TrackPair]

class OnboardingSelectionData(BaseModel):
    user_id: int = Field(..., description="사용자 ID")
    items: List[int] = Field(..., description="사용자가 선택한 트랙 ID 목록")

class MissingTrack(BaseModel):
    track_name: str
    artists_name: str
    playlist_name: str
    genres: List[str]
    length: int
    listeners: int

class PlaylistRequest(BaseModel):
    user_id: int
    tag: List[int]
    exists: List[int]
    missing: List[MissingTrack]

class PlaylistResponse(BaseModel):
    track_id: int
    track_name: str
    track_img_url: str
    artists: List[Dict[str, str]]
    description: str