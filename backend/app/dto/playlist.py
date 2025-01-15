from pydantic import BaseModel
from typing import Optional

class Playlist(BaseModel):
    playlist_id: str
    playlist_name: str
    playlist_img_url: Optional[str] = None

class Artist(BaseModel):
    artist_name: Optional[str]

class Track(BaseModel):
    track_id: str
    track_name: str
    track_img_url: Optional[str] = None
    artists: list[Artist]

class InsertTrack(BaseModel):
    track_name: str
    artists: list[Artist]

class InsertTrackRequest(BaseModel):
    items: list[InsertTrack]