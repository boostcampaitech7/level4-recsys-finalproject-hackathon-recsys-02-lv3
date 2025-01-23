from pydantic import BaseModel

class LoggingPlaylist(BaseModel):
    user_id: int
    track_id: int
    process: str
    action: str
    timestamp: str

class PlaylistLog(BaseModel):
    track_id: int
    process: str
    action: str

class LoggingPlaylistRequest(BaseModel):
    user_id: int
    items: list[PlaylistLog]
