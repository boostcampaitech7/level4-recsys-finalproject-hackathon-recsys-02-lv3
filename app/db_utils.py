import psycopg2
from psycopg2.extras import execute_values

from config import DB_CONFIG

def get_postgresql_connection():
    """PostgreSQL 연결 설정"""
    conn = psycopg2.connect(
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"]
    )
    return conn


def calculate_cosine_similarity(candidate_track_ids, playlist_track_ids, cur):
   
    query = """
    WITH candidate_tracks AS (
        SELECT track_id, track_emb
        FROM track_embedding
        WHERE track_id = ANY(%s)
    ),
    playlist_tracks AS (
        SELECT track_id, track_emb
        FROM track_embedding
        WHERE track_id = ANY(%s)
    )
    SELECT 
        c.track_id as candidate_track_id,
        p.track_id as playlist_track_id,
        CASE 
            WHEN (c.track_emb <#> c.track_emb) < 0 OR (p.track_emb <#> p.track_emb) < 0 THEN 0
            ELSE (c.track_emb <#> p.track_emb) / 
                (sqrt(c.track_emb <#> c.track_emb) * sqrt(p.track_emb <#> p.track_emb))
        END as cosine_similarity
    FROM 
        candidate_tracks c
        CROSS JOIN playlist_tracks p
    ORDER BY 
        cosine_similarity DESC;
    """
    
    cur.execute(query, (candidate_track_ids, playlist_track_ids))
    results = cur.fetchall()
    
    similarity_dict = {
        (row[0], row[1]): row[2] 
        for row in results
    }
    
    return similarity_dict
