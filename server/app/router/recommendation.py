import httpx
import re
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
        
@router.post("/test/onboarding", response_model=list[Onboarding])
async def get_onboarding_dummy_data(onboadingRequest: OnboardingRequest, db: Session = Depends(get_db)):
    find_user = db.query(User).filter(User.user_id == onboadingRequest.user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    pairs = [TrackIdPair(item1=i, item2=i+1) for i in range(1, 21, 2)]
    items1_id = [pair.item1 for pair in pairs]
    items2_id = [pair.item2 for pair in pairs]
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
    results1 = db.execute(query, {"track_id": items1_id}).fetchall()
    results2 = db.execute(query, {"track_id": items2_id}).fetchall()
    items1 = [Onboarding(
        track_id=result[0],
        track_name=result[1],
        artists=[Artist(artist_name=result[2]).dict()],
        track_img_url=result[3],
    ).dict() for result in results1]
    items2 = [Onboarding(
        track_id=result[0],
        track_name=result[1],
        artists=[Artist(artist_name=result[2]).dict()],
        track_img_url=result[3],
    ).dict() for result in results2]
    return JSONResponse(status_code=200, content={"items1":items1, "items2":items2})
        
@router.get("/playlists/{playlist_id}/tracks", response_model=list[Recommendation])
async def get_recommendation_by_playlists(playlist_id: str, user_id: int = Query(...), playlist_name: Optional[str] = None, \
                             db: Session = Depends(get_db)):
    find_user = db.query(User).filter(User.user_id == user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    response = await spotify_service._make_request(
        method='GET',
        url=f"{setting.SPOTIFY_API_URL}/playlists/{playlist_id}/tracks",
        user=find_user,
        db=db
    )
    items = response["items"]
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
    logger.info(tracks)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{setting.MODEL_API_URL}/playlist",
            json=PlaylistRecommendation(user_id=user_id, items=tracks).dict()
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
            recommendations = []
            results = db.execute(query, {"track_id": response.json()['items']}).fetchall() 
            for result in results:
                recommendations.append(Recommendation(
                    track_id=result[0],
                    track_name=result[1],
                    artists=[Artist(artist_name=artist).dict() for artist in result[2].split(' & ')],
                    track_img_url=result[3],
                ).dict())
            return JSONResponse(status_code=200, content=recommendations)
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

@router.get("/test/lastfm/{playlist_id}", response_model=RecommendationRequest)
async def get_metadata_based_playlist(playlist_id: str, user_id: int = Query(...), playlist_name: Optional[str] = None, \
                             db: Session = Depends(get_db)):
    find_user = db.query(User).filter(User.user_id == user_id).first()
    if not find_user:
        raise HTTPException(status_code=404, detail="cannot find user")
    
    response = await spotify_service._make_request(
        method='GET',
        url=f"{setting.SPOTIFY_API_URL}/playlists/{playlist_id}/tracks",
        user=find_user,
        db=db
    )
    items = response["items"]
    exists = []
    missing = []
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
        track_name = item["track"]["name"]
        result = db.execute(query, {"artist_name": artist_name, "track_name": track_name}).fetchone()
        if result:
            exists.append(result[0])
        else:
            async with httpx.AsyncClient() as client:
                meta_data = await client.get(
                    f"{setting.LASTFM_API_URL}&api_key={setting.LASTFM_API_KEY}"
                    f"&artist={artist_name}&track={track_name}"
                )
                track = meta_data.json()["track"]
                logger.info(track["toptags"])
            missing.append(TrackMetaData(
                track_name=track["name"],
                artist_name=track["artist"]["name"],
                playlist_name=playlist_name,
                genres=[tag["name"] for tag in track["toptags"]["tag"]],
                length=int(track["duration"]),
                listeners=int(track["listeners"])
            ).dict())

    return JSONResponse(status_code=200, content=RecommendationRequest(user_id=user_id, exists=exists, missing=missing).dict())

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
