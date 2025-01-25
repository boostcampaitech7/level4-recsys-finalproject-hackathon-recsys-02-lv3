upsert_sql = """
CREATE TABLE IF NOT EXISTS user_embedding (
    user_id SERIAL PRIMARY KEY,
    user_emb vector(64)
);

UPDATE user_embedding u
SET user_emb = t.user_emb
FROM temp_user_embeddings t
WHERE u.user_id = t.user_id;

INSERT INTO user_embedding (user_id, user_emb)
SELECT t.user_id, t.user_emb
FROM temp_user_embeddings t
LEFT JOIN user_embedding u ON t.user_id = u.user_id
WHERE u.user_id IS NULL;

DROP TABLE temp_user_embeddings;
"""
