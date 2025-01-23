from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY
from app.config.settings import Settings

# SQLAlchemy Base 클래스 정의
Base = declarative_base()
setting = Settings()

# 사용자 테이블 정의
class User_Emb(Base):
    __tablename__ = "user_embedding"

    user_id = Column(Integer, primary_key=True)
    user_emb = Column(ARRAY(Integer))

# 데이터베이스 연결
engine = create_engine(setting.EMBEDDING_DATABASE_URL)

# SQLAlchemy 세션 생성
EmbeddingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# DB 테이블 생성
Base.metadata.create_all(bind=engine)