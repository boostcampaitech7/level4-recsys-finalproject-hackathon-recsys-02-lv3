from db.database_postgres import User
from db.database_embedding import User_Emb

class UserService:
    @staticmethod
    def get_or_create_user(postgres_db, spotify_id, access_token, refresh_token):
        user = postgres_db.query(User).filter(User.spotify_id == spotify_id).first()
        if user:
            user.access_token = access_token
            user.refresh_token = refresh_token
        else:
            user = User(spotify_id=spotify_id, access_token=access_token, refresh_token=refresh_token)
            postgres_db.add(user)
        postgres_db.commit()
        postgres_db.refresh(user)
        return user

    @staticmethod
    def user_has_embedding(embedding_db, user_id):
        return embedding_db.query(User_Emb).filter(User_Emb.user_id == user_id).first() is not None