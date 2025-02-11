import sys
import os
import asyncio
import json
from typing import List

import numpy as np
import pandas as pd
import redis
import psycopg2
from tqdm import tqdm
from omegaconf import OmegaConf

from fastapi import FastAPI, HTTPException

from models.model import get_model, inference, compute_ratings
from schemas.schemas import (
    RecommendationRequest, 
    RecommendationResponse, 
    OnboardingData, 
    OnboardingResponse, 
    OnboardingSelectionData, 
    TrackPair, 
    PlaylistRequest, 
    PlaylistResponse
)
from utils.db_utils import get_postgresql_connection, get_onboarding_track_ids

from BiEncoder.src.rerank import recommend_songs
from BiEncoder.src.train import load_model as load_song_query_model
from BiEncoder.src.post_hoc import (
    get_candidate_meta, 
    get_selected_meta, 
    generate_preference_reason, 
    generate_npmi_score, 
    generate_popularity_score, 
    generate_description
)

app = FastAPI()
model, item_embs = None, None
song_encoder, query_encoder, scaler, data_songs = None, None, None, None


r = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True  
)


def load_redis_data():
    """
    PostgreSQL의 user_track 테이블에서 각 track_id에 대해 user_id 배열을 가져와
    Redis에 "track:<track_id>" 키로 JSON 문자열로 저장합니다.
    """
    conn = None
    try:
        conn = get_postgresql_connection(dbname="postgres")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT track_id, array_agg(user_id) as user_vector
                FROM user_track
                GROUP BY track_id;
            """)
            for track_id, user_vector in tqdm(cur, desc="Loading data to Redis"):
                key = f"track:{track_id}"
                r.set(key, json.dumps(user_vector))
    except Exception as e:
        print(f"Error loading Redis data: {e}")
    finally:
        conn.close()


@app.on_event("startup")
async def load_model():
    """FastAPI 시작 시 비동기적으로 모델 로딩"""
    global model, item_embs, song_encoder, query_encoder, scaler, data_songs
    print("Loading redis...")
    #load_redis_data()
    print("--------Redis complete---------")

    loop = asyncio.get_event_loop()

    lightgcn_task = loop.run_in_executor(None, get_model)
    biencoder_task = loop.run_in_executor(
        None, load_song_query_model, "./BiEncoder/config.yaml", "./BiEncoder/src/song_query_model.pt"
    )

    (model, item_embs), (song_encoder, query_encoder, scaler, data_songs) = await asyncio.gather(
        lightgcn_task, biencoder_task
    )

    print("All models loaded successfully")


@app.post("/onboarding", response_model=OnboardingResponse)
async def onboarding(request: OnboardingData):
    try:
        postgres_conn = get_postgresql_connection(dbname="postgres")
        tags_tuple = tuple(request.tags)

        query = """
            SELECT UNNEST(track::bigint[]) AS track_id
            FROM tag
            WHERE tag_id IN %s;
        """
        with postgres_conn.cursor() as cursor:
            cursor.execute(query, (tags_tuple,))
            rows = cursor.fetchall()
        postgres_conn.close()

        collected_track_ids = [row[0] for row in rows]
        total_list = tuple(set(collected_track_ids))

        if not total_list:
            raise HTTPException(status_code=404, detail="No tracks found for the given tags.")

        embedding_conn = get_postgresql_connection(dbname="embedding")
        similarity_query = f"""
            SELECT
                a.track_id AS track_id1,
                b.track_id AS track_id2,
                a.track_emb <=> b.track_emb AS cosine_similarity
            FROM track_embedding a
            CROSS JOIN track_embedding b
            WHERE a.track_id = ANY(%s)
              AND b.track_id = ANY(%s)
              AND a.track_id < b.track_id
            ORDER BY cosine_similarity;
        """

        df = pd.read_sql(similarity_query, embedding_conn, params=(list(total_list), list(total_list)))
        embedding_conn.close()

        used_tracks = set()
        pair_list = []
        for _, row in df.iterrows():
            track_id1 = int(row["track_id1"])
            track_id2 = int(row["track_id2"])
            if track_id1 not in used_tracks and track_id2 not in used_tracks:
                pair_list.append(TrackPair(item1=track_id1, item2=track_id2))
                used_tracks.add(track_id1)
                used_tracks.add(track_id2)
            if len(pair_list) >= 10:
                break

        if len(pair_list) < 10:
            raise HTTPException(status_code=404, detail="Not enough track pairs found.")

        return OnboardingResponse(items=pair_list)

    except Exception as e:
        print(f"Error in /onboarding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/onboarding/select", response_model=RecommendationResponse)
async def onboarding_select(request: OnboardingSelectionData):
    """
    사용자가 선택한 트랙들의 임베딩을 평균 내어 사용자 임베딩으로 저장 후, 추천 트랙 100개 반환.
    """

    #onboarding_10track[request.user_id]=request.items
    
    if not request.items:
        raise HTTPException(status_code=400, detail="No track IDs provided.")

    conn = get_postgresql_connection()
    cur = conn.cursor()

    try:
        query = "SELECT track_id, track_emb FROM track_embedding WHERE track_id = ANY(%s);"
        cur.execute(query, (request.items,))
        rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No track embeddings found.")

        embeddings = []
        for row in rows:
            track_emb = row[1]
    
            if isinstance(track_emb, str):
                try:
                    track_emb = json.loads(track_emb)
                except Exception as ex:
                    try:
                        track_emb = [float(x) for x in track_emb.strip('[]').split(',')]
                    except Exception as ex2:
                        cur.close()
                        conn.close()
                        raise HTTPException(status_code=500, detail=f"Error converting track_emb to float: {ex2}")
            embeddings.append(np.array(track_emb, dtype=float))
        
        if not embeddings:
            raise HTTPException(status_code=404, detail="No valid embeddings retrieved.")

        avg_embedding = np.mean(embeddings, axis=0) 

        upsert_query = """
            INSERT INTO user_embedding (user_id, user_emb)
            VALUES (%s, %s)
            ON CONFLICT (user_id)
            DO UPDATE SET user_emb = EXCLUDED.user_emb;
        """
        avg_emb_str = json.dumps(avg_embedding.tolist())
        cur.execute(upsert_query, (request.user_id, avg_emb_str))
        conn.commit()

        ratings = compute_ratings(avg_embedding, item_embs)  
        print(f"Ratings calculated: {ratings.shape}")
        top_100_track_ids = inference(ratings, top_k=100)  
        print(f"Top 100 recommended track IDs: {top_100_track_ids}")

        return RecommendationResponse(items=top_100_track_ids)

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error in /onboarding/select: {str(e)}")

    finally:
        cur.close()
        conn.close()


@app.post("/playlist", response_model=List[PlaylistResponse])
async def playlist(request: PlaylistRequest):
    """
    사용자 데이터를 기반으로 추천 트랙 목록 반환.
    """
    try:
        loop = asyncio.get_event_loop()

        def fetch_user_embedding():
            conn = get_postgresql_connection(dbname="embedding")
            cur = conn.cursor()
            query = "SELECT user_emb FROM user_embedding WHERE user_id = %s;"
            cur.execute(query, (request.user_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if not row:
                raise HTTPException(status_code=404, detail="User embedding not found in database.")
            user_emb = row[0]
            return json.loads(user_emb) if isinstance(user_emb, str) else np.array(user_emb, dtype=float)

        user_emb_task = loop.run_in_executor(None, fetch_user_embedding)
        selected_track_ids_task = loop.run_in_executor(None, get_onboarding_track_ids, request.user_id)

        user_emb, selected_track_ids = await asyncio.gather(user_emb_task, selected_track_ids_task)

        def compute_candidates():
            ratings = compute_ratings(user_emb, item_embs)
            top_100_track_ids = inference(ratings, top_k=100)
            return top_100_track_ids

        candidates_task = loop.run_in_executor(None, compute_candidates)

        df = pd.read_feather("fast_result.feather")
        track_with_id_df = df[["track", "track_id"]]
        config_path = "./BiEncoder/config.yaml"
        config = OmegaConf.load(config_path)

        candidates_track_ids = await candidates_task

        candidate_meta_task = loop.run_in_executor(None, get_candidate_meta, df, candidates_track_ids)
        recommend_task = loop.run_in_executor(
            None, recommend_songs, song_encoder, query_encoder, request.dict(), candidates_track_ids,
            "./BiEncoder/config.yaml", "./BiEncoder/src/song_query_model.pt", scaler, data_songs
        )

        candidate_meta_df, recommended_track_ids = await asyncio.gather(candidate_meta_task, recommend_task)
        candidate_meta_df.columns = ['track_id', 'track', 'listeners', 'img_url', 'artists', 'genres']

        reranked_track_ids = recommended_track_ids

        genre_pref = []
        for tag in request.tag:
            genre_pref.extend(config.clusters[tag]) 

        selected_meta_df = get_selected_meta(df, selected_track_ids)
        artist_pref = list(set(selected_meta_df['artist'].to_numpy()))

        df = generate_preference_reason(candidate_meta_df, genre_pref, artist_pref)
        df = generate_popularity_score(df)
        df = generate_npmi_score(track_with_id_df, df, selected_track_ids, candidates_track_ids, r)
        df["description"] = df.apply(generate_description, axis=1)

        df_indexed = df.set_index('track_id')
        sorted_df = df_indexed.loc[reranked_track_ids].reset_index()

        response_list = []
        for _, row in sorted_df.iterrows():
            response_list.append({
                "track_id": row["track_id"],
                "track_name": row["track"],
                "track_img_url": row["img_url"],
                "artists": [{"artist_name": artist} for artist in row["artists"] if artist],
                "description": row["description"] 
            })

        return response_list

    except Exception as e:
        print(f"[ERROR] /playlist: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
