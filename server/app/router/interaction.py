from pymongo import MongoClient
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import certifi
import pytz
import logging
from datetime import datetime
from app.config.settings import Settings
from app.dto.interaction import Log, Interactions, SelectedTrack
from app.service.model_service import ModelService

setting = Settings()
model_service = ModelService()
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
async def get_interaction_by_result(interactions: Interactions):
    '''
    플레이리스트 추천 결과를 바탕으로 생성된 상호작용 로그 적재
    '''
    items = interactions.items
    for item in items:
        log = Log(
            user_id=interactions.user_id, 
            track_id=item.track_id, 
            process=item.process, 
            action=item.action, 
            timestamp=get_seoul_timestamp() 
        )
        interaction_collection.insert_one(log.dict())
    return JSONResponse(status_code=200, content={"message":"logging complete"})

@router.post("/onboarding/select")
async def get_interaction_by_onboarding(interactions: Interactions):
    '''
    온보딩 과정에서 생성된 상호작용 로그 적재 및 유저 임베딩 생성
    '''
    tracks = []
    for item in interactions.items:
        log = Log(
            user_id=interactions.user_id, 
            track_id=item.track_id, 
            process=item.process, 
            action=item.action, 
            timestamp=get_seoul_timestamp()
        )
        if item.action == "positive":
            tracks.append(item.track_id)
        interaction_collection.insert_one(log.dict())
    logger.info(f"selected track: {tracks}")

    response = await model_service.make_request(
        method='POST',
        url="/onboarding/select",
        json=SelectedTrack(user_id=interactions.user_id, items=tracks).dict()
    )

    return JSONResponse(status_code=200, content={"message":"Successfully created user embedding"})