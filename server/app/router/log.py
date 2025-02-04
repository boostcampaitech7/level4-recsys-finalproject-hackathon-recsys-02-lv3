from pymongo import MongoClient
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import certifi
import pytz
import httpx
import logging
from datetime import datetime
from app.config.settings import Settings
from app.dto.log import LoggingPlaylist, LoggingPlaylistRequest
from app.dto.recommendation import PlaylistRecommendation

setting = Settings()
logger = logging.getLogger("uvicorn")

def get_seoul_timestamp():
    utc_now = datetime.now(pytz.utc)
    seoul_timezone = pytz.timezone('Asia/Seoul')
    seoul_now = utc_now.astimezone(seoul_timezone)
    return seoul_now.isoformat()

ca = certifi.where()

client = MongoClient(
    setting.MONGODB_DATABASE_URL,
    tlsCAFile=ca
)

db = client.get_database()
interaction_collection = db.interaction

router = APIRouter()

@router.post("/log/playlist")
async def get_onboarding(loggingPlaylistRequest: LoggingPlaylistRequest):
    items = loggingPlaylistRequest.items
    for item in items:
        log = LoggingPlaylist(
            user_id=loggingPlaylistRequest.user_id, 
            track_id=item.track_id, 
            process=item.process, 
            action=item.action, 
            timestamp=get_seoul_timestamp() 
        )
        interaction_collection.insert_one(log.dict())
    return JSONResponse(status_code=200, content={"message":"logging complete"})

@router.post("/onboarding/select")
async def get_onboarding(loggingPlaylistRequest: LoggingPlaylistRequest):
    items = loggingPlaylistRequest.items
    tracks = []
    for item in items:
        log = LoggingPlaylist(
            user_id=loggingPlaylistRequest.user_id, 
            track_id=item.track_id, 
            process=item.process, 
            action=item.action, 
            timestamp=get_seoul_timestamp()
        )
        if item.action == "positive":
            tracks.append(item.track_id)
        interaction_collection.insert_one(log.dict())
    logger.info(f"selected track: {tracks}")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{setting.MODEL_API_URL}/onboarding/select",
            json=PlaylistRecommendation(user_id=loggingPlaylistRequest.user_id, items=tracks).dict()
        )

    if response.status_code == 200:
        return JSONResponse(status_code=200, content={"message":"success"})
    else:
        raise HTTPException(status_code=response.status_code, detail=response.json())