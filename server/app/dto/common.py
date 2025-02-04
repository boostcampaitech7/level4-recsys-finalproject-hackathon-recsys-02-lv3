from pydantic import BaseModel
from typing import Optional

class Artist(BaseModel):
    artist_name: Optional[str]