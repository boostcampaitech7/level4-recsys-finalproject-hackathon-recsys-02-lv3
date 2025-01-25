upsert_sql = """
CREATE TABLE IF NOT EXISTS track_embedding (
    track_id SERIAL PRIMARY KEY,
    track_emb vector(64)
);

UPDATE track_embedding t 
SET track_emb = te.track_emb
FROM temp_track_embeddings te 
WHERE t.track_id = te.track_id;

INSERT INTO track_embedding (track_id, track_emb)
SELECT te.track_id, te.track_emb
FROM temp_track_embeddings te
LEFT JOIN track_embedding t ON t.track_id = te.track_id
WHERE t.track_id IS NULL;

DROP TABLE temp_track_embeddings;
"""
