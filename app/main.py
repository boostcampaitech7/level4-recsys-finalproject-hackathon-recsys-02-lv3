from fastapi import FastAPI, HTTPException
from model import get_model, inference, compute_ratings
from schemas import RecommendationRequest, RecommendationResponse, OnboardingData
from get_embs import get_user_embedding
from db_utils import get_postgresql_connection, calculate_cosine_similarity
import asyncio
import json


app = FastAPI()
model, item_embs = None, None

# TODO : 태그에 대한 임베딩을 db에서 가져오는 과정 필요
TOP50_BY_TAG_PATH = "./data/top50_by_tag.json"  # 태그별 임베딩 JSON 파일 경로
with open(TOP50_BY_TAG_PATH, "r", encoding="utf-8") as f:
        top50_by_tag = json.load(f)

#TODO : PostgreSQL 연결해서 user_id, user_embedding 저장
user_data = {} # 임시 딕셔너리


@app.on_event("startup")
async def load_model():
    """FastAPI 시작 시 비동기적으로 모델 로딩"""
    global model, item_embs
    print("Loading model...")
    loop = asyncio.get_event_loop()
    model, item_embs = await loop.run_in_executor(None, get_model)
    print("Model loaded successfully")


@app.post("/onboarding")
async def onboarding(data: OnboardingData):
    """온보딩에서 태그를 받아서 임베딩 생성 후 저장"""
    try:
        user_embedding = get_user_embedding(data.tags, top50_by_tag)  # get_user_embedding을 통해 임베딩 계산
        user_data[data.user_id] = user_embedding  # 서버 메모리에 저장
        return {"message": "Data received successfully", "user_id": data.user_id, "tags": data.tags}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

# TODO : 추천결과 생성 최적화
@app.post("/playlist", response_model=RecommendationResponse)
async def playlist(request: RecommendationRequest):
    """추천 요청 시 저장된 user_embedding을 이용해 추천"""
    try:
        user_embedding = user_data.get(request.user_id)
        if user_embedding is None:
            raise HTTPException(status_code=404, detail="User embedding not found")

        # 로깅 추가
        print(f"User embedding for {request.user_id}: {user_embedding}")

        # 임베딩을 사용하여 추천 계산
        ratings = compute_ratings(user_embedding, item_embs)  # user_embedding을 사용해 ratings 계산
        print(f"Ratings calculated: {ratings.shape}")

        recommended_track_ids = inference(ratings)
        print(f"Recommended track IDs: {recommended_track_ids}")
        print(f"Request items: {request.items}")

        if not recommended_track_ids:
            raise HTTPException(status_code=404, detail="No recommended track IDs found")

        conn = get_postgresql_connection()
        cur = conn.cursor()

        similarities = calculate_cosine_similarity(recommended_track_ids, request.items, cur)

        if similarities:
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_10_track_ids = []
            seen_tracks = set()
            for (candidate_track_id, _), sim in sorted_similarities:
                if candidate_track_id not in seen_tracks:
                    top_10_track_ids.append(candidate_track_id)
                    seen_tracks.add(candidate_track_id)
                if len(top_10_track_ids) == 10:
                    break
        else:
            raise HTTPException(status_code=401, detail="No similarities found")

        cur.close()
        conn.close()

        return RecommendationResponse(items=top_10_track_ids)
    except Exception as e:
        print(f"Error during recommendation: {str(e)}")  # 예외 메시지 로깅
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
