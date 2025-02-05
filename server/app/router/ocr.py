import asyncio
import logging
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from difflib import get_close_matches
from app.service.spotify_service import SpotifyService
from app.service.lastfm_service import LastfmService
from app.service.model_service import ModelService
from app.service.ocr_service import OCRService
from app.config.settings import Settings
from app.dto.common import Recommendation, RecommendationRequest, InsertTrackRequest
from app.dto.ocr import OCRTrack, OCRRecommendation
from db.database_postgres import PostgresSessionLocal, User

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
lastfm_service = LastfmService()
model_service = ModelService()
ocr_service = OCRService()

@router.post("/playlist/image", response_model=list[OCRTrack])
async def upload_playlist_image(
    user_id: int = Form(...),
    image: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    '''
    플레이리스트 캡쳐 이미지에서 ocr api를 통해 곡명과 아티스트명 추출
    '''

    # 유저 정보 확인 
    find_user = db.query(User).filter(User.user_id == user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    track_artist = await ocr_service.make_request(image)  
    tracks = []
    for track, artist in track_artist:
        tracks.append(OCRTrack(
            track_name=track,
            artist_name=artist
        ).dict())
    return JSONResponse(status_code=200, content=tracks)
    
@router.post("/playlist/image/tracks", response_model=Recommendation)
async def get_recommendation_by_image(ocrRecommendation: OCRRecommendation, db: Session = Depends(get_db)):
    '''
    플레이리스트 캡쳐 이미지 기반으로 추천 결과 생성
    '''
    find_user = db.query(User).filter(User.user_id == ocrRecommendation.user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    track_artist_list = [(ocr_service.replace_text(item.track_name), ocr_service.replace_text(item.artist_name)) for item in ocrRecommendation.items]

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

    # DB에서 찾은 트랙 ID 정리
    track_dict = {}
    for track_id, track, artist, listeners in db_results:
        key = (ocr_service.normalize_text(track), ocr_service.normalize_text(artist))
        if key not in track_dict or listeners > track_dict[key][1]:  
            track_dict[key] = (track_id, listeners)

    exists = []
    missing_requests = []

    for track, artist in track_artist_list:
        normalized_track = ocr_service.normalize_text(track)
        normalized_artist = ocr_service.normalize_text(artist)

        # 유사한 제목 찾기
        possible_matches = [
            (key, track_dict[key][1])
            for key in track_dict.keys()
            if get_close_matches(normalized_track, [key[0]], n=1, cutoff=0.6)  # 유사도 60% 이상
            and get_close_matches(normalized_artist, [key[1]], n=1, cutoff=0.6)
        ]

        if possible_matches:
            best_match = max(possible_matches, key=lambda x: x[1])[0]  # 가장 높은 listeners 수를 가진 key 선택
            exists.append(track_dict[best_match][0])
        else:
            missing_requests.append((artist, track))  # 못 찾은 트랙 저장
    
    logger.info(exists)

    # Last.fm API 병렬 요청
    missing_metadata = await asyncio.gather(*(lastfm_service.fetch_metadata(ocr_service.remove_text(artist), ocr_service.remove_text(track)) for artist, track in missing_requests))
    missing = [meta for meta in missing_metadata if meta is not None]

    logger.info(missing)

    response = await model_service.make_request(
        method='POST',
        url='/playlist',
        json=RecommendationRequest(user_id=ocrRecommendation.user_id, exists=exists, missing=missing).dict()
    )

    return JSONResponse(status_code=200, content=response)

@router.post("/playlist/create")
async def create_playlist(tracks: InsertTrackRequest, user_id: int = Query(...), db: Session = Depends(get_db)):
    '''
    유저가 선택한 트랙을 담은 새로운 플레이리스트 생성
    '''
    # 유저 정보 확인
    find_user = db.query(User).filter(User.user_id == user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    track_uris = []
    for track in tracks.items:
        query = f'track:"{track.track_name}" artist:"{track.artists[0].artist_name}"'
        response = await spotify_service.make_request(
            method='GET',
            url="/search",
            user=find_user,
            db=db,
            params={"q":query, "type":"track", "limit":3}
        )
        items = response['tracks']['items']
        if items:
            track_uris.append(items[0]['uri'])
    if track_uris:
        response = await spotify_service.make_request(
            method='POST',
            url=f"/users/{find_user.spotify_id}/playlists",
            user=find_user,
            db=db, 
            json={"name":"new playlist"}
        )
        playlist_id = response["id"]
        response = await spotify_service.make_request(
            method='POST',
            url=f"/playlists/{playlist_id}/tracks",
            user=find_user,
            db=db,
            json={"uris":track_uris}
        )
        return JSONResponse(status_code=200, content={"message":"Playlist created successfully"})
    else:
        return JSONResponse(status_code=200, content={"message":"cannot find tracks"})
