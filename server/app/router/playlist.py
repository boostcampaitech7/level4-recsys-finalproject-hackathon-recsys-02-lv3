import logging
import asyncio
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import Optional
from app.service.spotify_service import SpotifyService
from app.service.lastfm_service import LastfmService
from app.service.model_service import ModelService
from app.dto.playlist import Playlist
from app.dto.common import Recommendation, RecommendationRequest, InsertTrackRequest
from db.database import SessionLocal, User

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

router = APIRouter()
spotify_service = SpotifyService()
lastfm_service = LastfmService()
model_service = ModelService()
logger = logging.getLogger("uvicorn")

@router.get("/users/{user_id}/playlists", response_model=list[Playlist])
async def get_user_playlists(user_id: int, db: Session = Depends(get_db)):
    '''
    사용자가 스포티파이에 보유하고 있는 플레이리스트 목록 가져오기
    '''

    # 유저 정보 확인
    find_user = db.query(User).filter(User.user_id == user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    # Spotify API 요청
    response = await spotify_service.make_request(
        method='GET',
        url=f"/users/{find_user.spotify_id}/playlists",
        user=find_user,
        db=db
    )

    items = response["items"]
    playlists = [Playlist(
        playlist_id=item["id"], 
        playlist_name=item["name"],
        playlist_img_url=item["images"][0]["url"] if item["images"] else None
    ).dict()for item in items]

    return JSONResponse(status_code=200, content={"message":"playlists loaded successfully", "items":playlists})
        
@router.get("/playlists/{playlist_id}/tracks", response_model=list[Recommendation])
async def get_recommendation_by_playlist(
    playlist_id: str,
    user_id: int = Query(...),
    playlist_name: Optional[str] = " ",
    db: Session = Depends(get_db)
):
    '''
    스포티파이 플레이리스트에 있는 트랙들을 기준으로 추천 결과 생성
    '''

    # 유저 정보 확인 
    find_user = db.query(User).filter(User.user_id == user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    # Spotify API 요청
    response = await spotify_service.make_request(
        method='GET',
        url=f"/playlists/{playlist_id}/tracks",
        user=find_user,
        db=db
    )

    items = response["items"]
    track_artist_list = [
        (
            item["track"]["name"],
            item["track"]["artists"][0]["name"].split("&")[0]  # 첫 번째 아티스트만 사용
        )
        for item in items
    ]

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

    track_dict = {}
    for track_id, track, artist, listeners in db_results:
        key = (track, artist)
        if key not in track_dict or listeners > track_dict[key][1]:  
            track_dict[key] = (track_id, listeners)  # 여러 개일 경우 listeners가 가장 많은 트랙 선택

    exists = []
    missing_requests = []
    
    for track, artist in track_artist_list:
        key = (track, artist)
        if key in track_dict:
            exists.append(track_dict[key][0])
        else:
            missing_requests.append((artist, track))
    logger.info(find_user.tag)

    # Last.fm API 병렬 요청
    missing_metadata = await asyncio.gather(*(lastfm_service.fetch_metadata(artist, track, playlist_name) for artist, track in missing_requests))
    missing = [meta for meta in missing_metadata if meta is not None]

    response = await model_service.make_request(
        method='POST',
        url="/playlist",
        json=RecommendationRequest(user_id=user_id, tag=find_user.tag, exists=exists, missing=missing).dict()
    )
    return JSONResponse(status_code=200, content=response)

@router.post("/playlists/{playlist_id}/tracks")
async def insert_track_to_playlist(playlist_id: str, tracks: InsertTrackRequest, user_id: int = Query(...), \
                             db: Session = Depends(get_db)):
    '''
    유저가 선택한 트랙을 해당 플레이리스트에 삽입 
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
            url=f"/playlists/{playlist_id}/tracks",
            user=find_user,
            db=db,
            json={"uris":track_uris}
        )
        return JSONResponse(status_code=200, content={"message":"Tracks added successfully"})
    else:
        return JSONResponse(status_code=200, content={"message":"cannot find tracks"})