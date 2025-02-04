import httpx
import re
import asyncio
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from difflib import get_close_matches
from app.dto.recommendation import OCRTrack, OCRRecommendation
from app.service.spotify_service import SpotifyService
from app.config.settings import Settings
from app.dto.common import TrackMetaData, RecommendationRequest
from db.database_postgres import PostgresSessionLocal, User
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
spotify_service = SpotifyService()

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
    
@router.post("/playlist/image/tracks", response_model=RecommendationRequest)
async def get_recommendation_by_image(ocrRecommendation: OCRRecommendation, db: Session = Depends(get_db)):
    find_user = db.query(User).filter(User.user_id == ocrRecommendation.user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    def replace_ellipses(text: str):
        return text.replace('...', '%').strip()
    
    def remove_ellipses(text: str):
        return text.replace('%', '')
    
    track_artist_list = [(replace_ellipses(item.track_name), replace_ellipses(item.artist_name)) for item in ocrRecommendation.items]

    query = text("""
        SELECT t.track_id, t.track, a.artist, t.listeners
        FROM track t
        JOIN track_artist ta ON ta.track_id = t.track_id
        JOIN artist a ON ta.artist_id = a.artist_id
        WHERE 
            (a.artist LIKE ANY (ARRAY[:artist_names])) 
            AND (t.track LIKE ANY (ARRAY[:track_names]));
    """)

    track_names = [track for track, artist in track_artist_list]
    artist_names = [artist.split(' & ')[0] for track, artist in track_artist_list]

    db_results = db.execute(query, {"artist_names": artist_names, "track_names": track_names}).fetchall()

    def normalize_title(title):
        title = title.lower().strip()
        title = title.replace(" ", "")  # 공백 제거
        return title

    # DB에서 찾은 트랙 ID 정리 (listeners가 가장 높은 트랙 저장)
    track_dict = {}
    for track_id, track, artist, listeners in db_results:
        key = (normalize_title(track), normalize_title(artist))
        if key not in track_dict or listeners > track_dict[key][1]:  
            track_dict[key] = (track_id, listeners)  # listeners 수가 가장 많은 트랙 선택

    exists = []
    missing_requests = []

    for track, artist in track_artist_list:
        normalized_track = normalize_title(track)
        normalized_artist = normalize_title(artist)

        # 유사한 제목 찾기
        possible_matches = [
            (key, track_dict[key][1])  # (트랙 key, listeners 수)
            for key in track_dict.keys()
            if get_close_matches(normalized_track, [key[0]], n=1, cutoff=0.6)  # 유사도 60% 이상
            and get_close_matches(normalized_artist, [key[1]], n=1, cutoff=0.6)
        ]

        if possible_matches:
            # 🔹 listeners 수가 가장 높은 트랙 선택
            best_match = max(possible_matches, key=lambda x: x[1])[0]  # 가장 높은 listeners 수를 가진 key 선택
            exists.append(track_dict[best_match][0])
        else:
            missing_requests.append((artist, track))  # 못 찾은 트랙 저장
    # Last.fm API 병렬 요청
    async def fetch_metadata(artist, track):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{setting.LASTFM_API_URL}&api_key={setting.LASTFM_API_KEY}&artist={remove_ellipses(artist)}&track={remove_ellipses(track)}"
            )
            if response.status_code == 200:
                track_data = response.json().get("track", {})
                return TrackMetaData(
                    track_name=track_data.get("name", track),
                    artists_name=track_data.get("artist", {}).get("name", artist),
                    genres=[tag["name"] for tag in track_data.get("toptags", {}).get("tag", [])],
                    length=int(track_data.get("duration", 0)),
                    listeners=int(track_data.get("listeners", 0))
                ).dict()
        return None

    # 병렬로 API 요청 실행
    missing_metadata = await asyncio.gather(*(fetch_metadata(artist, track) for artist, track in missing_requests))
    missing = [meta for meta in missing_metadata if meta is not None]

    logger.info(f"exsits: {exists}")
    logger.info(f"missing: {missing}")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{setting.MODEL_API_URL}/playlist",
            json=RecommendationRequest(user_id=ocrRecommendation.user_id, exists=exists, missing=missing).dict(), 
            timeout=60
        )
        if response.status_code == 200:
            return JSONResponse(status_code=200, content=response.json())
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())