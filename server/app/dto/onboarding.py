from pydantic import BaseModel
from typing import Optional
from app.dto.common import Artist

class OnboardingRequest(BaseModel):
    user_id: int
    tags: list[int]

class Onboarding(BaseModel):
    track_id: int
    track_name: str
    track_img_url: Optional[str] = None
    artists: list[Artist]
    tags: list[str] = ["태그"]