from pydantic import BaseModel
from app.dto.common import Artist
from typing import Optional, Union

class Playlist(BaseModel):
    playlist_id: str
    playlist_name: str
    playlist_img_url: Optional[str] = None

class Track(BaseModel):
    track_id: Union[str, int]
    track_name: str
    track_img_url: Optional[str] = None
    artists: list[Artist]

class InsertTrack(BaseModel):
    track_name: str
    artists: list[Artist]

class InsertTrackRequest(BaseModel):
    items: list[InsertTrack]