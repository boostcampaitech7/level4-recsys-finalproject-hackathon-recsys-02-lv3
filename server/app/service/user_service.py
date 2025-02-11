from db.database import User
from db.embedding import EmbeddingSessionLocal, User_Emb

class UserService:
    @staticmethod
    def get_or_create_user(db, spotify_id, access_token, refresh_token):
        user = db.query(User).filter(User.spotify_id == spotify_id).first()
        if user:
            user.access_token = access_token
            user.refresh_token = refresh_token
        else:
            user = User(spotify_id=spotify_id, access_token=access_token, refresh_token=refresh_token)
            db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def user_has_embedding(user_id):
        db = EmbeddingSessionLocal()  # 데이터베이스 세션을 직접 생성
        try:
            return db.query(User_Emb).filter(User_Emb.user_id == user_id).first() is not None
        finally:
            db.close()