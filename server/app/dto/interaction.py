from pydantic import BaseModel

class Log(BaseModel):
    user_id: int
    track_id: int
    process: str
    action: str
    timestamp: str

class OneInteraction(BaseModel):
    track_id: int
    process: str
    action: str

class Interactions(BaseModel):
    user_id: int
    items: list[OneInteraction]

class SelectedTrack(BaseModel):
    user_id: int
    items: list[int]
