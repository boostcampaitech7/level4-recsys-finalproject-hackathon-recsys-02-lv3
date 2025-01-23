import httpx
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.dto.recommendation import OnboardingRequest, PlaylistRecommendation
from app.config.settings import Settings
from app.dto.playlist import Artist, Track
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

@router.post("/onboarding")
async def get_onboarding(onboadingRequest: OnboardingRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{setting.MODEL_API_URL}/onboarding",
            json=onboadingRequest.dict()
        )
        if response.status_code == 200:
            return JSONResponse(status_code=200, content={"message":"success"})
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())
        
@router.get("/playlists/{playlist_id}/tracks", response_model=list[Track])
async def get_recommendation_by_playlists(playlist_id: str, user_id: int = Query(...), \
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
            tracks = []
            for item in items:
                query = text("""
                SELECT t.track_id
                FROM track t
                JOIN track_artist ta ON ta.track_id = t.track_id
                JOIN artist a ON ta.artist_id = a.artist_id
                WHERE LOWER(REPLACE(a.artist, ' ', '')) = LOWER(REPLACE(:artist_name, ' ', ''))
                AND LOWER(REPLACE(t.track, ' ', '')) = LOWER(REPLACE(:track_name, ' ', ''));
                """)
                artist_name = item["track"]["artists"][0]["name"].split("&")[0]
                result = db.execute(query, {"artist_name": artist_name, "track_name": item["track"]["name"]}).fetchone()
                if result:
                    tracks.append(result[0])
            response = await client.post(
                f"{setting.MODEL_API_URL}/playlist",
                json=PlaylistRecommendation(user_id=user_id, items=tracks).dict()
            )
            if response.status_code == 200:
                recommendations = []
                for track in response.json()['items']:
                    query = text("""
                    SELECT t.track, 
                    STRING_AGG(a.artist, ' & ' ORDER BY a.artist) AS artist_names,
                    t.img_url
                    FROM track t
                    JOIN track_artist ta ON ta.track_id = t.track_id
                    JOIN artist a ON ta.artist_id = a.artist_id
                    WHERE t.track_id = :track_id
                    GROUP BY t.track_id, t.track, t.img_url;
                    """)
                    result = db.execute(query, {"track_id": track}).fetchone()
                    recommendations.append(Track(
                        track_id=track, 
                        track_name=result[0],
                        artists=[Artist(artist_name=result[1]).dict()],
                        track_img_url=result[2],
                    ).dict())
                return JSONResponse(status_code=200, content=recommendations)
            else:
                raise HTTPException(status_code=response.status_code, detail=response.json())
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())
        
@router.get("/test/playlists/{playlist_id}/tracks", response_model=list[Track])
async def test_get_recommendation_by_playlists_100(playlist_id: str, user_id: int = Query(...), \
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
            recommendations = []
            index = [i for i in range(1, 101)]
            query = text("""
                    SELECT 
                        t.track_id,
                        t.track AS track_name, 
                        STRING_AGG(a.artist, ' & ' ORDER BY a.artist) AS artist_names,
                        t.img_url
                    FROM track t
                    JOIN track_artist ta ON ta.track_id = t.track_id
                    JOIN artist a ON ta.artist_id = a.artist_id
                    WHERE t.track_id = ANY(:track_id)  -- track_id 배열
                    GROUP BY t.track, t.img_url, t.track_id
                    ORDER BY ARRAY_POSITION(:track_id, t.track_id);
                    """)
            results = db.execute(query, {"track_id": index}).fetchall() 
            for result in results:
                recommendations.append(Track(
                    track_id=result[0],
                    track_name=result[1],
                    artists=[Artist(artist_name=artist).dict() for artist in result[2].split(' & ')],
                    track_img_url=result[3],
                ).dict())

            return JSONResponse(status_code=200, content=recommendations)
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())