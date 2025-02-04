from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    CLIENT_ID: str
    CLIENT_SECRET: str
    REDIRECT_URI: str
    SPOTIFY_AUTHENTICATION_URL: str
    SPOTIFY_API_URL: str
    POSTGRES_DATABASE_URL: str
    EMBEDDING_DATABASE_URL: str
    MONGODB_DATABASE_URL: str
    MODEL_API_URL: str
    UPSTAGE_OCR_API_URL: str
    UPSTAGE_API_KEY: str
    LASTFM_API_KEY: str
    LASTFM_API_URL: str
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    FRONT_BASE_URL: str

    class Config:
        env_file = "app/config/.env"
        env_file_encoding = 'utf-8'
