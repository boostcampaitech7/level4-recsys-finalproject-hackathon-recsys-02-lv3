from pydantic import BaseModel, Field
from typing import List, Any

class RecommendationRequest(BaseModel):
    user_id: int
    items: List[int]

class RecommendationResponse(BaseModel):
    items: List[int]

class OnboardingData(BaseModel):
    user_id: int = Field(..., description="사용자 ID")
    tags: List[str] = Field(..., description="사용자 선호 태그 목록")

class OnboardingResponse(BaseModel):
    items: List[dict]

class OnboardingSelectionData(BaseModel):
    user_id: int = Field(..., description="사용자 ID")
    items: List[int] = Field(..., description="사용자가 선택한 트랙 ID 목록")
