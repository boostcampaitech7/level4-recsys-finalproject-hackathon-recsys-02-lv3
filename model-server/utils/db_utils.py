import psycopg2
from psycopg2.extras import execute_values
from pymongo import MongoClient

from .config import DB_CONFIG, MONGODB_URI

def get_postgresql_connection(dbname="embedding"):
    """PostgreSQL 연결 설정"""
    conn = psycopg2.connect(
        dbname=dbname,
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"]
    )
    return conn

def get_mongodb_connection():
    client = MongoClient(MONGODB_URI)
    db = client['spotify']
    collection = db['interaction']

    return collection

def get_onboarding_track_ids(user_id: int):
    collection = get_mongodb_connection()

    cursor = (
        collection.find({"user_id": user_id, "process": "onboarding", "action":  "positive"})
        .sort("timestamp", -1) 
        .limit(10)  
    )

    track_ids = [doc["track_id"] for doc in cursor]
    return track_ids
