from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLAlchemy Base 클래스 정의
Base = declarative_base()

# 사용자 테이블 정의
class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    spotify_id = Column(String, nullable=True)
    access_token = Column(String)
    refresh_token = Column(String)

# SQLite 데이터베이스 연결
DATABASE_URL = "sqlite:///./test.db"  # DB 파일 경로
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# SQLAlchemy 세션 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# DB 테이블 생성
Base.metadata.create_all(bind=engine)
