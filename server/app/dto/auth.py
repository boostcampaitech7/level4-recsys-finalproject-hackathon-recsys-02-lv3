from pydantic import BaseModel

class SpotifyTokenRequest(BaseModel):
    code: str
    redirect_uri: str
    grant_type: str = "authorization_code"

class SpotifyRefreshingTokenRequest(BaseModel):
    refresh_token: str
    grant_type: str = "refresh_token"
    