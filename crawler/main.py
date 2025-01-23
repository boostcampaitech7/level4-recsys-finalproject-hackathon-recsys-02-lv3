import pandas as pd
import os

from crawl import get_info, merge_with_benchmark_data
from transform_benchmark_to_table import transform_track, transform_artist, tranform_track_artist, transform_genre, tranform_track_genre, transform_playlist, transform_track_playlist, transform_user, transform_user_track
from csv_to_db import load_to_db
from utils import Directory

if __name__ == "__main__":
    get_info()
    merge_with_benchmark_data()

    data_path = os.path.join(Directory.DOWNLODAD_DIR, "benchmark/total_data")
    df = pd.read_csv(data_path)

    print("transform to table")
    track_table = transform_track(df)
    artist_table = transform_artist(df)

    tranform_track_artist(df, artist_table, track_table)
    genre_table = transform_genre(df)
    tranform_track_genre(df, genre_table, track_table)
    transform_playlist(df)
    transform_track_playlist(df)

    user_table = transform_user(df)
    transform_user_track(df,track_table, user_table)

    print("Transform to table form successfully!")
    table_list = ['track', 'artist', 'track_artist', 'genre', 'track_genre', 'playlist', 'track_playlist']

    for table in table_list:
        load_to_db(table)