import json
import redis
from fastapi import HTTPException
from app.config.settings import Settings
from app.dto.onboarding import Onboarding
from app.dto.common import Artist

setting = Settings()
redis_client = redis.Redis(host=setting.REDIS_HOST, port=setting.REDIS_PORT, password=setting.REDIS_PASSWORD, decode_responses=True)

def create_redirect_url(user, embedding, user_img_url=None):
    base_url = setting.FRONT_BASE_URL
    if embedding:
        redirect_url = f"{base_url}/home?user_id={user.user_id}"
    else:
        redirect_url = f"{base_url}/onboarding?user_id={user.user_id}"
    if user_img_url:
        redirect_url += f"&user_img_url={user_img_url}"
    return redirect_url

def fetch_tracks_from_redis(track_ids):
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