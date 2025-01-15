from pydantic import BaseModel
from typing import Optional

class SpotifyTokenRequest(BaseModel):
    code: str
    redirect_uri: str
    grant_type: str = "authorization_code"

class SpotifyRefreshingTokenRequest(BaseModel):
    refresh_token: str
    grant_type: str = "refresh_token"

class GetUserResponse(BaseModel):
    message: str
    user_id: int
    user_img_url: Optional[str]