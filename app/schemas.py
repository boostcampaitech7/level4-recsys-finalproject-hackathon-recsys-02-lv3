from pydantic import BaseModel, Field
from typing import List

class RecommendationRequest(BaseModel):
    user_id: int
    items: List[int]

class RecommendationResponse(BaseModel):
    items: List[int]

class OnboardingData(BaseModel):
    user_id: int = Field(..., description="사용자 ID")
    tags: List[str] = Field(..., description="사용자 선호 태그 목록")
