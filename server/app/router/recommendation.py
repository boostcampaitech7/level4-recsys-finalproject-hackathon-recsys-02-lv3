import httpx
import re
import redis
import json
import asyncio
from fastapi import APIRouter, HTTPException, Query, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.dto.recommendation import OnboardingRequest, PlaylistRecommendation, TrackIdPair, OCRTrack, OCRRecommendation
from app.utils.spotify_api_service import SpotifyApiService
from app.config.settings import Settings
from app.dto.recommendation import Onboarding, Artist, Recommendation, TrackMetaData, RecommendationRequest
from db.database_postgres import PostgresSessionLocal, User
from typing import Optional
import logging

def get_db():
    db = PostgresSessionLocal()
    try:
        yield db
    finally:
        db.close()

logger = logging.getLogger("uvicorn")
router = APIRouter()
setting = Settings()
spotify_service = SpotifyApiService()
r = redis.Redis(host=setting.REDIS_HOST, port=setting.REDIS_PORT, password=setting.REDIS_PASSWORD, decode_responses=True)
        
@router.post("/onboarding", response_model=list[Onboarding])
async def get_onboarding(onboadingRequest: OnboardingRequest, db: Session = Depends(get_db)):
    find_user = db.query(User).filter(User.user_id == onboadingRequest.user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    items1 = []
    items2 = []
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{setting.MODEL_API_URL}/onboarding",
            json=onboadingRequest.dict()
        )
        if response.status_code == 200:
            pairs = response.json()["items"]
            items1_id = [pair["item1"] for pair in pairs]
            items2_id = [pair["item2"] for pair in pairs]
            def fetch_tracks_from_redis(track_ids, redis_client):
                track_data_list = redis_client.mget([str(track_id) for track_id in track_ids])
                items = []

                for track_data in track_data_list:
                    if not track_data:
                        raise HTTPException(status_code=404, detail="Track not found in Redis")
                    
                    track_info = json.loads(track_data)
                    items.append(Onboarding(
                        track_id=track_info.get("track_id", 0),
                        track_name=track_info.get("track_name", ""),
                        track_img_url=track_info.get("track_img_url", ""),
                        artists=[Artist(artist_name=artist) for artist in track_info.get("artist", "")], 
                        tags=[track_info.get("tag", "")]
                    ).dict())

                return items

            # items1, items2 처리
            items1 = fetch_tracks_from_redis(items1_id, r)
            items2 = fetch_tracks_from_redis(items2_id, r)
            return JSONResponse(status_code=200, content={"items1":items1, "items2":items2})
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())
        
@router.get("/playlists/{playlist_id}/tracks", response_model=list[Recommendation])
async def get_metadata_based_playlist(
    playlist_id: str,
    user_id: int = Query(...),
    playlist_name: Optional[str] = " ",
    db: Session = Depends(get_db)
):
    find_user = db.query(User).filter(User.user_id == user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    # Spotify API 요청
    response = await spotify_service._make_request(
        method='GET',
        url=f"{setting.SPOTIFY_API_URL}/playlists/{playlist_id}/tracks",
        user=find_user,
        db=db
    )
    items = response["items"]

    # Track & Artist 리스트 준비
    track_artist_list = [
        (
            item["track"]["name"],
            item["track"]["artists"][0]["name"].split("&")[0]  # 첫 번째 아티스트만 사용
        )
        for item in items
    ]

    # 한 번의 쿼리로 모든 트랙 검색
    query = text("""
        SELECT t.track_id, t.track, a.artist, t.listeners
        FROM track t
        JOIN track_artist ta ON ta.track_id = t.track_id
        JOIN artist a ON ta.artist_id = a.artist_id
        WHERE a.artist IN :artist_names
        AND t.track IN :track_names;
    """)

    track_names = [track.strip() for track, artist in track_artist_list]
    artist_names = [artist.strip() for track, artist in track_artist_list]

    db_results = db.execute(query, {"artist_names": tuple(artist_names), "track_names": tuple(track_names)}).fetchall()

    # DB에서 찾은 트랙 ID 정리
    track_dict = {}
    for track_id, track, artist, listeners in db_results:
        key = (track.lower().replace(" ", ""), artist.lower().replace(" ", ""))
        if key not in track_dict or listeners > track_dict[key][1]:  
            track_dict[key] = (track_id, listeners)  # listeners가 가장 많은 트랙 선택

    exists = []
    missing_requests = []
    
    for track, artist in track_artist_list:
        key = (track.lower().replace(" ", ""), artist.lower().replace(" ", ""))
        if key in track_dict:
            exists.append(track_dict[key][0])
        else:
            missing_requests.append((artist, track))

    # Last.fm API 병렬 요청
    async def fetch_metadata(artist, track):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{setting.LASTFM_API_URL}&api_key={setting.LASTFM_API_KEY}&artist={artist}&track={track}"
            )
            if response.status_code == 200:
                track_data = response.json().get("track", {})
                return TrackMetaData(
                    track_name=track_data.get("name", track),
                    artists_name=track_data.get("artist", {}).get("name", artist),
                    playlist_name=playlist_name,
                    genres=[tag["name"] for tag in track_data.get("toptags", {}).get("tag", [])],
                    length=int(track_data.get("duration", 0)),
                    listeners=int(track_data.get("listeners", 0))
                ).dict()
        return None

    # 병렬로 API 요청 실행
    missing_metadata = await asyncio.gather(*(fetch_metadata(artist, track) for artist, track in missing_requests))
    missing = [meta for meta in missing_metadata if meta is not None]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{setting.MODEL_API_URL}/playlist",
            json=RecommendationRequest(user_id=user_id, exists=exists, missing=missing).dict()
        )
        if response.status_code == 200:
            return JSONResponse(status_code=200, content=response.json())
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())

@router.post("/playlist/image", response_model=list[OCRTrack])
async def upload_playlist_image(
    user_id: int = Form(...),
    image: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    find_user = db.query(User).filter(User.user_id == user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    try:
        files = {"document": (image.filename, await image.read(), image.content_type)}
        headers = {
                "Authorization": f"Bearer {setting.UPSTAGE_API_KEY}"
            }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{setting.UPSTAGE_OCR_API_URL}",
                headers=headers,
                files=files
            )

            if response.status_code == 200:
                def remove_video(text: str):
                    return re.sub(r'동영상\s*·?\s*', '', text)
                
                def remove_leading_19(track: str):
                    return re.sub(r'^19\s*', '', track) 

                def extract_song_artist(text: str):
                    text = remove_video(text)
                    pattern = r'([^\n·]+)\s*·?\s*([^\n·]+)'
                    matches = re.findall(pattern, text)
                    matches = [(remove_leading_19(track), remove_leading_19(artist)) for track, artist in matches]
                    return matches
                result_text = response.json()["text"]
                track_artist = extract_song_artist(result_text)
                
                tracks = []
                for track, artist in track_artist:
                    tracks.append(OCRTrack(
                        track_name=track,
                        artist_name=artist
                    ).dict())
                return JSONResponse(status_code=200, content=tracks)
            else:
                raise HTTPException(status_code=response.status_code, detail=response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    

@router.post("/playlist/image/tracks", response_model=list[Recommendation])
async def get_recommendation_by_image(ocrRecommendation: OCRRecommendation, db: Session = Depends(get_db)):
    find_user = db.query(User).filter(User.user_id == ocrRecommendation.user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    def replace_ellipses(text: str):
        return text.replace('...', '%')
    
    logger.info(ocrRecommendation)
    track_artist = [(replace_ellipses(item.track_name), replace_ellipses(item.artist_name)) for item in ocrRecommendation.items]
    tracks = []
    for track, artist in track_artist:
        query = text("""
            SELECT
                t.track_id
            FROM track t
            JOIN track_artist ta ON ta.track_id = t.track_id
            JOIN artist a ON ta.artist_id = a.artist_id
            WHERE LOWER(REPLACE(a.artist, ' ', '')) LIKE LOWER(REPLACE(:artist_name, ' ', ''))
            AND LOWER(REPLACE(t.track, ' ', '')) LIKE LOWER(REPLACE(:track_name, ' ', ''));
        """)
        result = db.execute(query, {"artist_name": artist.split("&")[0].strip(), "track_name": track.strip()}).fetchone()
        if result:
            tracks.append(result[0])
    logger.info(tracks)
    recommendations = []
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{setting.MODEL_API_URL}/playlist",
            json=PlaylistRecommendation(user_id=ocrRecommendation.user_id, items=tracks).dict()
        )
        if response.status_code == 200:
            query = text("""
                SELECT 
                    t.track_id,
                    t.track AS track_name, 
                    STRING_AGG(DISTINCT a.artist, ' & ' ORDER BY a.artist) AS artist_names,
                    t.img_url
                FROM track t
                JOIN track_artist ta ON ta.track_id = t.track_id
                JOIN artist a ON ta.artist_id = a.artist_id
                WHERE t.track_id = ANY(:track_id)
                GROUP BY t.track, t.img_url, t.track_id
                ORDER BY ARRAY_POSITION(:track_id, t.track_id);
            """)
            results = db.execute(query, {"track_id": response.json()['items']}).fetchall() 
            for result in results:
                recommendations.append(Recommendation(
                    track_id=result[0],
                    track_name=result[1],
                    artists=[Artist(artist_name=artist).dict() for artist in result[2].split(' & ')],
                    track_img_url=result[3],
                ).dict())
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())

    return JSONResponse(status_code=200, content=recommendations)

@router.post("/test/lastfm/image", response_model=RecommendationRequest)
async def get_metadata_based_image(ocrRecommendation: OCRRecommendation, db: Session = Depends(get_db)):
    find_user = db.query(User).filter(User.user_id == ocrRecommendation.user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    def replace_ellipses(text: str):
        return text.replace('...', '%')
    
    def remove_ellipses(text: str):
        return text.replace('%', '')
    
    track_artist = [(replace_ellipses(item.track_name), replace_ellipses(item.artist_name)) for item in ocrRecommendation.items]
    exists = []
    missing = []
    for track, artist in track_artist:
        query = text("""
            SELECT
                t.track_id
            FROM track t
            JOIN track_artist ta ON ta.track_id = t.track_id
            JOIN artist a ON ta.artist_id = a.artist_id
            WHERE LOWER(REPLACE(a.artist, ' ', '')) LIKE LOWER(REPLACE(:artist_name, ' ', ''))
            AND LOWER(REPLACE(t.track, ' ', '')) LIKE LOWER(REPLACE(:track_name, ' ', ''));
        """)
        artist_name = artist.split("&")[0]
        result = db.execute(query, {"artist_name": artist_name, "track_name": track}).fetchone()
        if result:
            exists.append(result[0])
        else:
            async with httpx.AsyncClient() as client:
                meta_data = await client.get(
                    f"{setting.LASTFM_API_URL}&api_key={setting.LASTFM_API_KEY}"
                    f"&artist={remove_ellipses(artist_name).strip()}&track={remove_ellipses(track).strip()}"
                )
                logger.info(meta_data.json())
                if "error" not in meta_data.json():
                    track = meta_data.json()["track"]
                    missing.append(TrackMetaData(
                        track_name=track["name"],
                        artist_name=track["artist"]["name"],
                        genres=[tag["name"] for tag in track["toptags"]["tag"]],
                        length=int(track["duration"]),
                        listeners=int(track["listeners"])
                    ).dict())

    return JSONResponse(status_code=200, content=RecommendationRequest(user_id=ocrRecommendation.user_id, exists=exists, missing=missing).dict())
