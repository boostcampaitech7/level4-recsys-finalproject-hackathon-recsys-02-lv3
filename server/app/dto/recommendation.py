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
    