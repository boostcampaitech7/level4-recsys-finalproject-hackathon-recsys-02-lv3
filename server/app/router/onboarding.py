from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.dto.onboarding import OnboardingRequest, Onboarding
from app.service.model_service import ModelService
from app.utils.utils import fetch_tracks_from_redis
from db.database_postgres import PostgresSessionLocal, User
import logging

router = APIRouter()
model_service = ModelService()
logger = logging.getLogger("uvicorn")

def get_db():
    db = PostgresSessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/onboarding", response_model=list[Onboarding])
async def get_onboarding(onboadingRequest: OnboardingRequest, db: Session = Depends(get_db)):
    '''
    onboarding system 수행 시 선택할 후보 목록 반환 
    '''

    # 유저 정보 확인 
    find_user = db.query(User).filter(User.user_id == onboadingRequest.user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    items1 = []
    items2 = []
    logger.info(f"tags: {onboadingRequest.tags}") # 디버깅
    
    response = await model_service.make_request(
        method='POST',
        url='/onboarding',
        json=onboadingRequest.dict()
    )

    pairs = response["items"]
    items1_id = [pair["item1"] for pair in pairs]
    items2_id = [pair["item2"] for pair in pairs]

    items1 = fetch_tracks_from_redis(items1_id)
    items2 = fetch_tracks_from_redis(items2_id)

    return JSONResponse(status_code=200, content={"items1":items1, "items2":items2})