from sqlalchemy import create_engine
import pandas as pd

USER = 'root'
PASSWORD = '1234'
HOST = "34.64.106.144"
PORT = "3306"
DB = "spotify-dataset"

def get_chunk_size(row_length):
    if row_length > 10000000:     # 천만 건 이상 (track_genre)
        return 50000
    elif row_length > 1000000:    # 백만 건 이상 (track_playlist, track_artist)
        return 20000
    elif row_length > 100000:     # 십만 건 이상 (track, playlist)
        return 10000
    else:                        # 그 외 (artist, genre)
        return 5000
    

def load_to_db(table):
    # PostgreSQL connection string format
    connection_string = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}"
    db_connection = create_engine(connection_string)
    
    # CSV 파일 읽기
    df = pd.read_csv(f'./transform/benchmark/{table}_table.csv')
    row_length = len(df)
    chunksize = get_chunk_size(row_length)
    
    # PostgreSQL에 데이터 로드
    df.to_sql(
        name=table,  # 테이블 이름
        con=db_connection,
        if_exists='replace',  # 기존 테이블이 있으면 대체
        index=False,
        chunksize=chunksize,
        method='multi'  # PostgreSQL에서 더 효율적인 벌크 인서트를 위해 추가
    )
    
    print(f"Load {table} data to PostgreSQL Successfully!")