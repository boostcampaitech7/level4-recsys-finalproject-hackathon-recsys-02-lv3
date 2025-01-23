import pandas as pd
import ast
import os

from utils import Directory

def transform_track(df):
    track_table = df[['track', 'last_fm_url', 'listeners', 'length', 'introduction']]
    track_table.drop_duplicates(subset = 'last_fm_url', inplace = True)
    track_table.insert(0, 'track_id', range(1, len(track_table) + 1))
    track_table.reset_index(drop=True)

    path = os.path.join(Directory.TRANSFORM_DIR, 'benchmark/track_table.csv')
    track_table.to_csv(path, index=False)
    print("Transform track successfully!")

    return track_table

def transform_artist(df):
    tmp_artist_table = df[['artist']]
    tmp_artist_table['artist'] = tmp_artist_table['artist'].str.split('[&]').apply(lambda x: [name.strip() for name in x])
    tmp_artist_table = tmp_artist_table.explode("artist")
    artist_list = tmp_artist_table.artist.unique()
    artist_table = pd.DataFrame({"artist_id" : range(1, len(artist_list) + 1), "artist" : artist_list})
    
    path = os.path.join(Directory.TRANSFORM_DIR, 'benchmark/artist_table.csv')
    artist_table.to_csv(path, index=False)
    print("Transform artist successfully!")

    return artist_table


def tranform_track_artist(df, artist_table, track_table):
    artist_table = pd.read_csv("./transform/artist_table.csv")
    track_table = pd.read_csv("./transform/track_table.csv")

    track_artist_table = df[['track', 'artist', 'last_fm_url']]
    track_artist_table['artist'] = track_artist_table['artist'].str.split('[&]').apply(lambda x: [name.strip() for name in x])

    track_artist_table = track_artist_table.explode("artist")

    track_artist_table['artist_id'] = track_artist_table['artist'].map(artist_table.set_index('artist')['artist_id'])
    track_artist_table['track_id'] = track_artist_table['last_fm_url'].map(track_table.set_index('last_fm_url')['track_id'])

    track_artist_table.drop(['track', 'artist', 'last_fm_url'], axis = 1, inplace = True)
    track_artist_table.insert(0, 'track_artist_id', range(1, len(track_artist_table) + 1))
    track_artist_table.reset_index(drop=True)

    path = os.path.join(Directory.TRANSFORM_DIR, 'benchmark/track_artist_table.csv')
    track_artist_table.to_csv(path, index=False)
    print("Transform track_artist successfully!")


def transform_genre(df):
    tmp_genre_table = df[['genres']]
    tmp_genre_table['genres'] = tmp_genre_table['genres'].apply(ast.literal_eval)
    tmp_genre_table = tmp_genre_table.explode("genres")

    genre_list = tmp_genre_table['genres'].unique().tolist()
    genre_table = pd.DataFrame({'genre_id' : range(1, len(genre_list)+ 1), 'genre' : genre_list})

    path = os.path.join(Directory.TRANSFORM_DIR, 'benchmark/genre_table.csv')
    genre_table.to_csv(path, index=False)
    print("Transform genre successfully!")

    return genre_table

def tranform_track_genre(df, genre_table, track_table):
    genre_table = pd.read_csv("./transform/genre_table.csv")
    track_table = pd.read_csv("./transform/track_table.csv")

    track_genre_table = df[['track', 'last_fm_url', 'genres']]
    track_genre_table.rename(columns={'genres' : 'genre'}, inplace=True)

    track_genre_table['genre'] = track_genre_table['genre'].apply(ast.literal_eval)
    track_genre_table = track_genre_table.explode("genre")

    track_genre_table['genre'] = track_genre_table['genre'].map(genre_table.set_index("genre")['genre_id'])
    track_genre_table['track_id'] = track_genre_table['last_fm_url'].map(track_table.set_index("last_fm_url")['track_id'])

    track_genre_table.drop(['track', 'last_fm_url'], axis = 1, inplace = True)

    path = os.path.join(Directory.TRANSFORM_DIR, 'benchmark/track_genre_table.csv')
    track_genre_table.to_csv(path, index=False)
    print("Transform track_genre successfully!")


def transform_playlist(df):
    playlist_list = df['playlist'].unique().tolist()
    playlist_table = pd.DataFrame({'playlist_id' : range(1, len(playlist_list)+1), 'playlist' : playlist_list})

    path = os.path.join(Directory.TRANSFORM_DIR, 'benchmark/playlist_table.csv')
    playlist_table.to_csv(path, index=False)
    print("Transform playlist successfully!")


def transform_track_playlist(df):
    track_table = pd.read_csv("./transform/track_table.csv")
    playlist_table = pd.read_csv("./transform/playlist_table.csv")

    track_playlist_table = df[['track', 'last_fm_url', 'playlist']]
    track_playlist_table['track_id'] = track_playlist_table['last_fm_url'].map(track_table.set_index('last_fm_url')['track_id'])
    track_playlist_table['playlist_id'] = track_playlist_table['playlist'].map(playlist_table.set_index('playlist')['playlist_id'])
    track_playlist_table.drop(['track', 'last_fm_url', 'playlist'], axis=1, inplace=True)

    path = os.path.join(Directory.TRANSFORM_DIR, 'benchmark/track_playlist_table.csv')
    track_playlist_table.to_csv(path, index=False)
    print("Transform track_playlist successfully!")

def transform_user(df):
    user_table = df[['user_id']]
    spotify_id_list = user_table['user_id'].unique()
    user_table = pd.DataFrame({'user_id' : range(1, len(spotify_id_list)+1), 'spotify_id' : spotify_id_list})
    user_table['access_token'] = ''
    user_table['refresh_token'] = ''
    
    path = os.path.join(Directory.TRANSFORM_DIR, 'benchmark/users_table.csv')
    user_table.to_csv(path, index=False)
    print("Transform user successfully!")

    return user_table

def transform_user_track(df, track_table, user_table):
    user_track_table = df[['user_id', 'track', 'last_fm_url']]
    user_track_table['track_id'] = user_track_table['last_fm_url'].map(track_table.set_index('last_fm_url')['track_id'])
    user_track_table['user_org_id'] = user_track_table['user_id'].map(user_table.set_index('spotify_id')['user_id'])
    user_track_table = user_track_table[['user_org_id', 'track_id']]
    user_track_table.columns = ['user_id', 'track_id']
    user_track_table.insert(0, 'user_track_id', range(1, len(user_track_table) + 1))

    path = os.path.join(Directory.TRANSFORM_DIR, 'benchmark/user_track_table.csv')
    user_track_table.to_csv(path, index=False)
    print("Transform user_track successfully!")