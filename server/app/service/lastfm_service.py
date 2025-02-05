import httpx
from app.config.settings import Settings
from app.dto.common import TrackMetaData

class LastfmService():
    def __init__(self):
        self.setting = Settings()

    async def fetch_metadata(self, artist, track, playlist_name = " "):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.setting.LASTFM_API_URL}&api_key={self.setting.LASTFM_API_KEY}&artist={artist}&track={track}"
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