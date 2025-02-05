from pydantic import BaseModel

class OCRTrack(BaseModel):
    track_name: str
    artist_name: str

class OCRRecommendation(BaseModel):
    user_id: int
    items: list[OCRTrack]