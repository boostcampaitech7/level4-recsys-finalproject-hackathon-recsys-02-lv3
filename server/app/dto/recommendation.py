from pydantic import BaseModel

class OnboardingRequest(BaseModel):
    user_id: int
    tags: list[str]

class PlaylistRecommendation(BaseModel):
    user_id: int
    items: list[int]

class TrackIdPair(BaseModel):
    item1: int
    item2: int

class OCRTrack(BaseModel):
    track_name: str
    artist_name: str

class OCRRecommendation(BaseModel):
    user_id: int
    items: list[OCRTrack]