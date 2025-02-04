import httpx
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.config.settings import Settings
from app.utils.spotify_api_service import SpotifyApiService
from app.dto.playlist import Playlist, Artist, Track, InsertTrackRequest
from db.database_postgres import PostgresSessionLocal, User
import logging

def get_db():
    db = PostgresSessionLocal()
    try:
        yield db
    finally:
        db.close()

router = APIRouter()
setting = Settings()
logger = logging.getLogger("uvicorn")
spotify_service = SpotifyApiService()

@router.get("/users/{user_id}/playlists", response_model=list[Playlist])
async def get_user_playlists(user_id: int, db: Session = Depends(get_db)):
    find_user = db.query(User).filter(User.user_id == user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    response = await spotify_service._make_request(
        method='GET',
        url=f"{setting.SPOTIFY_API_URL}/users/{find_user.spotify_id}/playlists",
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
        
@router.post("/playlists/{playlist_id}/tracks")
async def insert_playlist_track(playlist_id: str, tracks: InsertTrackRequest, user_id: int = Query(...), \
                             db: Session = Depends(get_db)):
    find_user = db.query(User).filter(User.user_id == user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    track_uris = []
    for track in tracks.items:
        query = f'track:"{track.track_name}" artist:"{track.artists[0].artist_name}"'
        response = await spotify_service._make_request(
            method='GET',
            url=f"{setting.SPOTIFY_API_URL}/search",
            user=find_user,
            db=db,
            params={"q":query, "type":"track", "limit":3}
        )
        items = response['tracks']['items']
        if items:
            track_uris.append(items[0]['uri'])
            
    if track_uris:
        response = await spotify_service._make_request(
            method='POST',
            url=f"{setting.SPOTIFY_API_URL}/playlists/{playlist_id}/tracks",
            user=find_user,
            db=db,
            json={"uris":track_uris}
        )
        return JSONResponse(status_code=200, content={"message":"Tracks added successfully"})
    else:
        return JSONResponse(status_code=200, content={"message":"cannot find tracks"})