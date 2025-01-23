import httpx
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.config.settings import Settings
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

@router.get("/users/{user_id}/playlists", response_model=list[Playlist])
async def get_user_playlists(user_id: int, db: Session = Depends(get_db)):
    find_user = db.query(User).filter(User.user_id == user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{setting.SPOTIFY_API_URL}/users/{find_user.spotify_id}/playlists",
            headers={
                "Authorization":f"Bearer {find_user.access_token}"
            }
        )

        if response.status_code == 200:
            items = response.json()["items"]
            playlists = [Playlist(
                playlist_id=item["id"], 
                playlist_name=item["name"],
                playlist_img_url=item["images"][0]["url"] if item["images"] else None
            ).dict()for item in items]
            return JSONResponse(status_code=200, content={"message":"playlists loaded successfully", "items":playlists})
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())

@router.get("/test/playlists/{playlist_id}", response_model=list[Track])
async def get_playlist_track(playlist_id: str, user_id: int = Query(...), \
                             db: Session = Depends(get_db)):
    find_user = db.query(User).filter(User.user_id == user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{setting.SPOTIFY_API_URL}/playlists/{playlist_id}/tracks",
            headers={
                "Authorization":f"Bearer {find_user.access_token}"
            }
        )
        if response.status_code == 200:
            items = response.json()["items"]
            tracks = [Track(
                track_id=item["track"]["id"], 
                track_name=item["track"]["name"],
                track_img_url=item["track"]["album"]["images"][0]["url"] if item["track"]["album"]["images"] else None,
                artists=[Artist(artist_name=artist["name"]).dict() for artist in item["track"]["artists"]]
            ).dict() for item in items]
            return JSONResponse(status_code=200, content={"message":"playlists loaded successfully", "items":tracks})
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())
        
@router.post("/playlists/{playlist_id}/tracks")
async def insert_playlist_track(playlist_id: str, tracks: InsertTrackRequest, user_id: int = Query(...), \
                             db: Session = Depends(get_db)):
    find_user = db.query(User).filter(User.user_id == user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    headers = {"Authorization":f"Bearer {find_user.access_token}"}
    async with httpx.AsyncClient() as client:
        track_uris = []
        for track in tracks.items:
            logger.info(f"{tracks.items}")
            query = f'track:"{track.track_name}" artist:"{track.artists[0].artist_name}"'
            response = await client.get(
                f"{setting.SPOTIFY_API_URL}/search",
                headers=headers,
                params={"q":query, "type":"track", "limit":3}
            )
            if response.status_code == 200:
                items = response.json()['tracks']['items']
                if items:
                    track_uris.append(items[0]['uri'])
            else:
                raise HTTPException(status_code=response.status_code, detail=response.json())
        
        if track_uris:
            response = await client.post(
                f"{setting.SPOTIFY_API_URL}/playlists/{playlist_id}/tracks",
                headers=headers,
                json={"uris":track_uris}
            )
            if response.status_code == 200:
                return JSONResponse(status_code=200, content={"message":"Tracks added successfully"})
            else:
                raise HTTPException(status_code=response.status_code, detail=response.json())
        else:
            return JSONResponse(status_code=200, content={"message":"cannot find tracks"})