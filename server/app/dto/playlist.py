from pydantic import BaseModel
from typing import Optional

class Playlist(BaseModel):
    playlist_id: str
    playlist_name: str
    playlist_img_url: Optional[str] = None
