upsert_sql = """
UPDATE user_embedding u
SET user_id = t.user_id,
    user_emb = t.user_emb
FROM temp_user_embeddings t
WHERE u.user_org_id = t.user_org_id;

INSERT INTO user_embedding (user_org_id, user_id, user_emb)
SELECT t.user_org_id, t.user_id, t.user_emb
FROM temp_user_embeddings t
LEFT JOIN user_embedding u ON t.user_org_id = u.user_org_id
WHERE u.user_org_id IS NULL;

DROP TABLE temp_user_embeddings;
"""