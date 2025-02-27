from omegaconf import OmegaConf
import pandas as pd
from typing import Dict
import random
import numpy as np
from math import log
import json


def get_candidate_meta(df, candidate_track_id):
    '''
    후보곡의 메타정보 추출
    '''
    candidate_meta_df = df[df['track_id'].isin(candidate_track_id)]
    candidate_meta_df = candidate_meta_df.groupby(['track_id', 'track', 'listeners', 'img_url'], as_index=False).agg({
        'artist': lambda x: list(set(x)),  
        'genre': lambda x: list(set(x))  
    })
    
    return candidate_meta_df


def get_selected_meta(df, selected_track_id):
    '''
    선호 아티스트 추출 (온보딩 선택 트랙 기반)
    '''
    selected_meta_df = df[df['track_id'].isin(selected_track_id)][['track_id', 'artist']]
    return selected_meta_df


def generate_preference_reason(candidate_meta_df: pd.DataFrame, genre_pref: list, artist_pref: list) -> pd.DataFrame:
    '''
    preference 카테고리의 추천 이유 생성: 선호 장르/아티스트
    '''
    reasons = []
    keys = []

    for _, row in candidate_meta_df.iterrows():
        matched_genres = list(set(row['genres']) & set(genre_pref))
        matched_artists = list(set(row['artists']) & set(artist_pref))
        reason = None
        key = {}
        if matched_artists:
            reason = "artist"
            key['artist'] = [random.choice(matched_artists)]  
        elif matched_genres:
            reason = "genre"
            key['genre'] = [random.choice(matched_genres)]  
        key = key if key else None
        reasons.append(reason)
        keys.append(key)

    candidate_meta_df['reason'] = reasons
    candidate_meta_df['key'] = keys

    return candidate_meta_df


def get_track_vector(track_id, r):
    key = f"track:{track_id}"
    stored_vector = r.get(key)
    if stored_vector:
        return json.loads(stored_vector)
    return None

    

def calculate_npmi_from_vectors(selected_track, candidate_track, r, n_users=14959) -> float:
    strack_users = set(get_track_vector(selected_track, r))
    ctrack_users = set(get_track_vector(candidate_track, r))
    
    if not strack_users or not ctrack_users or selected_track == candidate_track:
        return float('-inf')

    intersection = len(strack_users.intersection(ctrack_users))
    if intersection == 0: return float('-inf')
    
    PMI = log((n_users * intersection) / (len(strack_users) * len(ctrack_users)))
    NPMI = PMI/(-log(intersection/n_users))
    return NPMI


def generate_npmi_score(track_with_id_df: pd.DataFrame, df: pd.DataFrame, selected_track_id: list, candidate_track_id: list, r) -> pd.DataFrame:
    """
    후보곡과 선택된 트랙 간의 NPMI 점수를 계산하여 DataFrame을 업데이트하는 함수.

    Args:
        df (pd.DataFrame): 후보곡 메타 정보가 담긴 DataFrame
        selected_track_id (list): 사용자가 선택한 트랙 ID 목록
        candidate_track_id (list): 추천 후보곡 ID 목록

    Returns:
        pd.DataFrame: NPMI 점수가 반영된 DataFrame
    """
    candidate_scores = []
    cnt, total = 0, 0
 
    # 'selected_track_id'에 해당하는 트랙 이름 조회
    track_name_dict = track_with_id_df[track_with_id_df['track_id'].isin(selected_track_id)].set_index('track_id')['track'].to_dict()

    # 각 후보곡에 대해 NPMI 점수 계산
    for candidate_id in candidate_track_id:
        c_score = {"track_id": candidate_id, "reason": 'co-occurrence', "key": None}
        score = float('-inf')

        for selected_id in selected_track_id:
            NPMI_score = calculate_npmi_from_vectors(selected_track=selected_id, candidate_track=candidate_id, r=r)
            total += 1
            if score < NPMI_score:
                score = NPMI_score
                track_name = track_name_dict.get(selected_id, "Unknown Track")  # 트랙 ID가 없으면 "Unknown Track"
                c_score["key"] = track_name
                cnt += 1

        candidate_scores.append(c_score)

    # df에서 업데이트할 행 찾아 reason, key 값 채우기
    for data in candidate_scores:
        tid, reason, key = data['track_id'], data['reason'], data['key']
        condition = (df['track_id'] == tid) & (df['reason'].isna()) 
        df.loc[condition, ['track_id', 'reason', 'key']] = [tid, reason, key]

    return df


def generate_popularity_score(df: pd.DataFrame) -> pd.DataFrame:
    '''
    popularity 카테고리의 추천 이유 생성
    '''
    ranges = [
        (215200, float('inf'), 1),     # 99th percentile
        (113600, 215200, 2),           # 98th percentile
        (74600, 113600, 3),            # 97th percentile
        (54500, 74600, 4),             # 96th percentile
        (42400, 54500, 5),             # 95th percentile
        (34300, 42400, 6),             # 94th percentile
        (28500, 34300, 7),             # 93th percentile
        (24200, 28500, 8),             # 92th percentile
        (20800, 24200, 9),             # 91th percentile
        (18100, 20800, 10),   ]        # 90th percentile
    
    empty_mask = df['reason'].isna()
    for idx in df[empty_mask].index:
        value = df.at[idx, 'listeners']
        if value < 18100:
            continue
        for lower, upper, key in ranges:
            if lower <= value < upper:
                df.at[idx, 'reason'] = 'popularity'
                df.at[idx, 'key'] = key
                break
    
    return df


def generate_description(row):
    '''
    추천 이유 설명문 생성
    '''
    if row["reason"] == "artist":
        artist_name = row["key"]["artist"][0]
        return f"선호하시는 {artist_name}의 노래예요"
    elif row["reason"] == "genre":
        genre_name = row["key"]["genre"][0]  
        return f"즐겨듣는 {genre_name} 노래예요"
    elif row["reason"] == "co-occurrence":        
        if row["key"] is None:
            return "" 
        try:
            track_name = row["key"]
            return f"[{track_name}] 트랙을 좋아한 분들이 함께 좋아했어요"
        except (IndexError, AttributeError): 
            return ""
    elif row["reason"] == "popularity":
        popularity_score = row["key"]
        return f"다른 분들이 많이 찾는 상위 {popularity_score}% 노래예요"
    return ""


if __name__ == "__main__":

    # 1) 데이터
    # 예시 INPUT
    tag_id = [1, 2, 12] # INPUT1 - 온보딩에서 선택한 태그
    selected_track_id=[1,2,3,4,5,6,7,8,9,10] # INPUT2 - 온보딩에서 선택한 트랙
    candidate_track_id=sorted(np.random.randint(0, 1000, 100).tolist()) # INPUT3 - 추천 후보 트랙

    # 더미 데이터
    

    # 후보곡의 메타 정보
    df = pd.read_feather("fast_result.feather")

    config_path = "../config.yaml"
    config = OmegaConf.load(config_path)
    candidate_meta_df = get_candidate_meta(df, candidate_track_id)
    candidate_meta_df.columns = ['track_id', 'track', 'listeners', 'img_url', 'artists', 'genres']

    # 2) 유저의 선호 아티스트, 장르 추출
    genre_pref = []
    for tag in tag_id:
        genre_pref.extend(config.clusters[tag]) # genres

    selected_meta_df = get_selected_meta(df, selected_track_id) 
    artist_pref = list(set(selected_meta_df['artist'].to_numpy())) # artists

    # 3) 추천 이유 생성 
    df = generate_preference_reason(candidate_meta_df, genre_pref, artist_pref) # preference
    df = generate_npmi_score(df, selected_track_id, candidate_track_id) # co-occurence
    df = generate_popularity_score(df) # popularity
    df["description"] = df.apply(generate_description, axis=1) # 설명문

    # 4) json 포맷 저장
    records = df.to_dict(orient="records")
    json_output = json.dumps(records, ensure_ascii=False, indent=4)
    with open("post_hoc_output.json", "w", encoding="utf-8") as f:
        f.write(json_output)