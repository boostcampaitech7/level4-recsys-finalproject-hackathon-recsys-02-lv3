from fastapi import FastAPI, HTTPException
from model import get_model, inference, compute_ratings
from schemas import RecommendationRequest, RecommendationResponse, OnboardingData, OnboardingResponse, OnboardingSelectionData
from get_embs import get_user_embedding
from db_utils import get_postgresql_connection, calculate_cosine_similarity
import asyncio
import json


app = FastAPI()
model, item_embs = None, None

# TODO : 태그에 대한 임베딩을 db에서 가져오는 과정 필요
# TOP50_BY_TAG_PATH = "./data/top50_by_tag.json"  # 태그별 임베딩 JSON 파일 경로
# with open(TOP50_BY_TAG_PATH, "r", encoding="utf-8") as f:
#         top50_by_tag = json.load(f)

with open("./data/tag_track_dic.json", "r", encoding="utf-8") as f:
    tag_track_dic = json.load(f)

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


# @app.post("/onboarding")
# async def onboarding(data: OnboardingData):
#     """온보딩에서 태그를 받아서 임베딩 생성 후 저장"""
#     try:
#         user_embedding = get_user_embedding(data.tags, top50_by_tag)  # get_user_embedding을 통해 임베딩 계산
#         user_data[data.user_id] = user_embedding  # 서버 메모리에 저장
#         return {"message": "Data received successfully", "user_id": data.user_id, "tags": data.tags}
#     except Exception as e:
#         raise HTTPException(status_code=422, detail=str(e))

#TODO: 연결풀 사용해서 DB 재사용
@app.post("/onboarding", response_model=OnboardingResponse)
async def onboarding(data: OnboardingData):
    try:
        # 1. 요청된 태그에 해당하는 트랙 ID들을 추출
        collected_track_ids = []
        for tag in request.tags:
            if tag in tag_track_dic:
                collected_track_ids.extend(tag_track_dic[tag])
        total_list = tuple(set(collected_track_ids))
        if not total_list:
            raise HTTPException(status_code=404, detail="No tracks found for given tags.")

        # 2. PostgreSQL에 연결하여 total_list 내의 모든 트랙 쌍에 대해 코사인 유사도를 계산
        conn = get_postgresql_connection()
        query = f"""
            SELECT
                a.track_id AS track_id1,
                b.track_id AS track_id2,
                a.track_emb <=> b.track_emb AS cosine_similarity
            FROM track_embedding a
            CROSS JOIN track_embedding b
            WHERE a.track_id IN {total_list}
              AND b.track_id IN {total_list}
              AND a.track_id < b.track_id
            ORDER BY cosine_similarity ASC;
        """
        # pandas를 이용해 SQL 쿼리 실행
        df = pd.read_sql(query, conn)
        conn.close()

        # 3. 중복 없이 10쌍의 트랙을 선택 (한 트랙이 두 번 이상 사용되지 않도록)
        used_tracks = set()
        pair_list = []
        for _, row in df.iterrows():
            track_id1 = int(row["track_id1"])
            track_id2 = int(row["track_id2"])
            if track_id1 not in used_tracks and track_id2 not in used_tracks:
                pair_list.append({"item1": track_id1, "item2": track_id2})
                used_tracks.add(track_id1)
                used_tracks.add(track_id2)
            if len(pair_list) >= 10:
                break

        if len(pair_list) < 10:
            raise HTTPException(status_code=404, detail="Not enough track pairs found.")

        return RecommendationResponse(items=pair_list)

    except Exception as e:
        print(f"Error in /playlist: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/onboarding/select", response_model=RecommendationResponse)
async def onboarding_select(request: OnboardingSelectionData):
    """
    사용자가 선택한 트랙들의 임베딩을 평균 내어 사용자 임베딩으로 저장 후, 추천 트랙 100개 반환.
    """
    if not request.items:
        raise HTTPException(status_code=400, detail="No track IDs provided.")

    conn = get_postgresql_connection()
    cur = conn.cursor()

    try:
        # 1. track_embedding 테이블에서 선택한 track_id의 임베딩 조회
        query = "SELECT track_id, track_emb FROM track_embedding WHERE track_id = ANY(%s);"
        cur.execute(query, (request.items,))
        rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No track embeddings found.")

        # 2. track_emb 리스트로 변환
        embeddings = [np.array(row[1]) for row in rows]  # `track_emb`가 `vector` 타입이므로 직접 변환 가능
        avg_embedding = np.mean(embeddings, axis=0)  # 평균 임베딩 계산

        # # 3. user_embedding 테이블에 업서트 (PostgreSQL vector 타입 사용)
        # upsert_query = """
        #     INSERT INTO user_embedding (user_id, user_emb)
        #     VALUES (%s, %s)
        #     ON CONFLICT (user_id)
        #     DO UPDATE SET user_emb = EXCLUDED.user_emb;
        # """
        # cur.execute(upsert_query, (request.user_id, avg_embedding.tolist()))
        # conn.commit()

        # 4. 모델을 이용하여 추천 트랙 100개 가져오기
        ratings = compute_ratings(avg_embedding, item_embs)  # 사용자 임베딩과 아이템 임베딩의 내적 계산
        top_100_track_ids = inference(ratings, top_k=100)  # 상위 100개 트랙 추천

        return RecommendationResponse(items=top_100_track_ids)

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error in /onboarding/select: {str(e)}")

    finally:
        cur.close()
        conn.close()


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
