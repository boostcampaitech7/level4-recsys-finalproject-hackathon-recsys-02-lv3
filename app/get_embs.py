import json
import random
import numpy as np

def get_user_embedding(tags: list, top50_by_tag: dict) -> np.ndarray:
    """
    태그를 기반으로 사용자 임베딩 계산
    :param tags: 사용자 태그 리스트
    :param top50_by_tag: JSON 파일 
    :return: 사용자 임베딩 (numpy 배열)
    """
    sampled_track_ids = []
    for tag in tags:
        if tag in top50_by_tag:
            tracks = top50_by_tag[tag]
            sampled_tracks = random.sample(tracks, min(3, len(tracks)))
            sampled_track_ids.extend([track["track_id"] for track in sampled_tracks])

    # track_embedding 추출
    embeddings = [
        track["track_embedding"]
        for tag in top50_by_tag.keys()
        for track in top50_by_tag[tag]
        if track["track_id"] in sampled_track_ids
    ]

    # user_embedding 계산
    if embeddings:
        user_embedding = np.mean(embeddings, axis=0)
        return user_embedding
    else:
        raise ValueError("No embeddings found for the selected tags.")
