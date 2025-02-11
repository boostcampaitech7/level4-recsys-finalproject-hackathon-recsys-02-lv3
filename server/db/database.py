from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import sessionmaker
from app.config.settings import Settings

# SQLAlchemy Base 클래스 정의
Base = declarative_base()
setting = Settings()

# 사용자 테이블 정의
class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    spotify_id = Column(String)
    access_token = Column(String)
    refresh_token = Column(String)
    tag = Column(ARRAY(Integer))

# 데이터베이스 연결
engine = create_engine(setting.POSTGRES_DATABASE_URL)

# SQLAlchemy 세션 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# DB 테이블 생성
Base.metadata.create_all(bind=engine)
