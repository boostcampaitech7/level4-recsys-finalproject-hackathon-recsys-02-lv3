import os
from itertools import combinations
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
import hashlib
import json
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class BERTTextEncoder(nn.Module):
    '''
    Freeze된 BERT 기반의 텍스트 임베딩을 수행하는 인코더 클래스

    Args:
        config (OmegaConf): 모델 설정 정보
    
    Attributes:
        tokenizer (AutoTokenizer): BERT 토크나이저
        bert (AutoModel): 사전 학습된 BERT 모델
        linear (nn.Linear): 최종 임베딩 차원으로 변환하는 선형 레이어
    '''
    def __init__(self, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.bert_pretrained)
        self.bert = AutoModel.from_pretrained(config.model.bert_pretrained)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(768, config.model.output_dim)
    
    def forward(self, texts):
        texts = [str(text) for text in texts]
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.linear.weight.device) for k, v in inputs.items()}
        
        outputs = self.bert(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        x = self.linear(cls_emb)
        x = F.relu(x)
        return x


class NumericEncoder(nn.Module):
    '''
    숫자형 데이터를 정규화해 처리하는 인코더 클래스

    Args:
        config (OmegaConf): 모델 설정 정보
    '''
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(2, 64)
        self.linear2 = nn.Linear(64, config.model.output_dim)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.relu(x)


class SongEncoder(nn.Module):
    '''
    곡의 다양한 메타데이터를 결합해 최종 임베딩을 생성하는 모델

    Args:
        config (OmegaConf): 모델 설정 정보
        playlist_info (dict, optional): 플레이리스트 관련 정보
        cluster_embeds (dict, optional): 클러스터 임베딩 정보
        clusters_dict (dict, optional): 클러스터 정보 딕셔너리
    '''

    def __init__(self, config, playlist_info=None, cluster_embeds=None, clusters_dict=None): 
        super().__init__()
        self.artist_encoder = PlaylistAwareArtistEncoder(playlist_info, output_dim=config.model.mha_embed_dim, cache_dir='./cache')
        self.track_encoder = BERTTextEncoder(config)
        self.playlist_encoder = BERTTextEncoder(config)

        self.genres_encoder = GenreClusterEncoder(clusters_dict, cluster_embeds, config)

        self.numeric_encoder = NumericEncoder(config)

        self.mha = nn.MultiheadAttention(embed_dim=config.model.mha_embed_dim, num_heads=config.model.mha_heads, batch_first=True)
        self.final_fc = nn.Linear(config.model.mha_embed_dim, config.model.final_dim)
        nn.init.xavier_uniform_(self.final_fc.weight)

    def forward(self, artists, tracks, playlists, listeners, lengths, genres):
        '''
        입력 데이터를 기반으로 곡 임베딩 생성

        Args:
            artists (List[str]): 아티스트 목록
            tracks (List[str]): 트랙 제목 목록
            playlists (List[str]): 플레이리스트 제목 목록
            listeners (torch.Tensor): 청취자 수
            lengths (torch.Tensor): 트랙 길이
            genres (List[List[str]]): 장르 목록

        Returns:
            torch.Tensor: 정규화된 곡 임베딩
        '''
            
        artist_emb = self.artist_encoder(artists)
        track_emb = self.track_encoder(tracks)
        playlist_emb = self.playlist_encoder(playlists)
        genres_emb = self.genres_encoder(genres)

        num_tensor = torch.tensor(list(zip(listeners, lengths)), dtype=torch.float32).to(artist_emb.device)
        numeric_emb = self.numeric_encoder(num_tensor)

        # Apply Multi-Head Attention
        features = torch.stack([artist_emb, track_emb, playlist_emb, genres_emb, numeric_emb], dim=1)
        
        attn_output, _ = self.mha(features, features, features)
        agg_vector = attn_output.mean(dim=1)

        final_emb = self.final_fc(agg_vector)

        return F.normalize(final_emb, p=2, dim=1)


class GenreEncoder(nn.Module):
    '''
    장르 정보를 기반으로 정규화된 임베딩을 생성하는 인코더 

    Args:
        config (OmegaConf): 모델 설정 정보
    '''

    def __init__(self, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.bert_pretrained)
        self.bert = AutoModel.from_pretrained(config.model.bert_pretrained)
        self.linear = nn.Linear(768, config.model.genre_embed_dim)

        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, genres_batch):
        if isinstance(genres_batch[0], list):
            texts = [" ".join(genres) for genres in genres_batch]
        else:
            texts = [str(genres) for genres in genres_batch]

        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.linear.weight.device) for k, v in inputs.items()}
        
        outputs = self.bert(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]

        x = self.linear(cls_emb)
        x = F.relu(x)
        return F.normalize(x, p=2, dim=1)


# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PlaylistAwareArtistEncoder(nn.Module):
    '''
    플레이리스트 정보를 활용하여 아티스트 임베딩을 생성하는 모델

    Args:
        playlist_info (dict): 플레이리스트와 아티스트 매핑 정보
        output_dim (int, optional): 출력 임베딩 차원
        cache_dir (str, optional): 캐시 디렉토리 경로
    '''
    def __init__(self, playlist_info, output_dim=64, cache_dir='./cache'):
        super().__init__()
        self.output_dim = output_dim
        self.artist2id = {}  # Mapping from artist name to ID
        self.id2artist = {}  # Mapping from ID to artist name
        self.playlist_info = playlist_info
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        # Generate graphh
        self.graph = self.get_or_create_graph()
        
        # Dynamic artist registration
        unique_artists = list(self.graph.nodes())
        self.artist2id = {artist: idx for idx, artist in enumerate(unique_artists)}
        self.id2artist = {idx: artist for artist, idx in self.artist2id.items()}
        
        # Initialize embedding layer on GPU
        self.embedding = nn.Embedding(len(unique_artists) + 100, output_dim).to(self.device)
        self.init_embeddings_with_graph_info()

    def get_cache_key(self):
        '''
        플레이리스트를 위한 MD5 hash고유 캐시 키 생성
        
        '''
        playlist_str = json.dumps(self.playlist_info, sort_keys=True)
        return hashlib.md5(playlist_str.encode()).hexdigest()

    def get_or_create_graph(self):
        '''
        캐싱된 그래프 로드(없다면 생성)
        
        '''
        cache_key = self.get_cache_key()
        cache_file = self.cache_dir / f"graph_{cache_key}.pkl"
        
        if cache_file.exists():
            print("🔄 Loading cached graph...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("🆕 Creating new graph...")
        graph = self.create_artist_graph_advanced()
        
        with open(cache_file, 'wb') as f:
            pickle.dump(graph, f)
        
        return graph

    def create_artist_graph_advanced(self):
        """
        플레이리스트 기반으로 아티스트 간 그래프 생성

        Returns:
            networkx.Graph: 아티스트 간의 그래프
        """
        print("🔍 Starting graph creation...")
        edge_weights = {}
        
        # Process playlists
        items = list(self.playlist_info.items())
        for playlist_name, artists in tqdm(items, desc="Processing playlists"):
            if not playlist_name.strip() or len(artists) < 2:
                continue
                
            cleaned_artists = [artist.strip() for artist in artists if artist and artist.strip()]
            if len(cleaned_artists) < 2:
                continue
                
            for artist1, artist2 in combinations(cleaned_artists, 2):
                edge = tuple(sorted([artist1, artist2]))
                edge_weights[edge] = edge_weights.get(edge, 0) + 1

        # Construct graph
        G = nx.Graph()
        G.add_weighted_edges_from(
            (artist1, artist2, weight)
            for (artist1, artist2), weight in edge_weights.items()
        )
        
        # Apply weight-based filtering
        weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
        weight_threshold = np.percentile(weights, 10)
        edges_to_remove = [(u, v) for (u, v, d) in G.edges(data=True) 
                          if d['weight'] < weight_threshold]
        G.remove_edges_from(edges_to_remove)
        
        print(f"🔍 Graph completed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def init_embeddings_with_graph_info(self):
        '''
        그래프의 주요 아티스트를 찾아 임베딩을 초기화
        '''
        print("📊 Initializing embeddings...")
        pagerank = nx.pagerank(self.graph)
        degree_cent = nx.degree_centrality(self.graph)
        
        # Allocate tensor for embeddings
        embedding_weights = torch.zeros(self.embedding.weight.shape, device=self.device)
        
        for artist, idx in tqdm(self.artist2id.items(), desc="Initializing embeddings"):
            centrality = (pagerank.get(artist, 0.5) + degree_cent.get(artist, 0.5)) / 2
            std = np.sqrt(2.0 / (self.graph.number_of_nodes() + self.output_dim))
            embedding_weights[idx] = torch.randn(self.output_dim, device=self.device) * std * centrality
        
        self.embedding.weight.data.copy_(embedding_weights)

    @torch.no_grad()
    def forward(self, artists_batch):
        """
        입력된 아티스트 배치에 대한 임베딩 반환

        Args:
            artists_batch (List[List[str]]): 트랙별 아티스트 리스트

        Returns:
            torch.Tensor: 아티스트 임베딩 벡터
        """
        all_artists = set()
        for track_artists in artists_batch:
            all_artists.update(track_artists)
        
        # Register new artists dynamically
        for artist in all_artists:
            self.register_new_artist(artist)
        
        # Compute batch embeddings on GPU
        batch_embeddings = []
        for track_artists in artists_batch:
            artist_indices = torch.tensor([self.artist2id[artist] for artist in track_artists], 
                                        device=self.device)
            track_embeddings = self.embedding(artist_indices)
            batch_embeddings.append(torch.mean(track_embeddings, dim=0))
        
        return torch.stack(batch_embeddings)

    def register_new_artist(self, artist):
        '''
        동적으로 새로운 아티스트 추가 
        
        '''
        if artist not in self.artist2id:
            with torch.no_grad():
                new_id = len(self.artist2id)
                self.artist2id[artist] = new_id
                self.id2artist[new_id] = artist
                
                # Expand embedding matrix if needed
                if new_id >= self.embedding.num_embeddings:
                    new_embedding = nn.Embedding(new_id + 100, self.output_dim).to(self.device)
                    new_embedding.weight.data[:self.embedding.num_embeddings] = self.embedding.weight.data
                    
                    # Initialize new embeddings
                    std = np.sqrt(2.0 / (self.graph.number_of_nodes() + self.output_dim))
                    new_embedding.weight.data[self.embedding.num_embeddings:] = \
                        torch.randn(new_embedding.num_embeddings - self.embedding.num_embeddings, 
                                  self.output_dim, device=self.device) * std
                    
                    self.embedding = new_embedding

    def to(self, device):
        """
        모델을 디바이스로 이동
        """
        self.device = device
        self.embedding = self.embedding.to(device)
        return super().to(device)


class DistilBertTextEncoder(torch.nn.Module):
    '''
    DistilBERT 기반의 텍스트 임베딩을 수행하는 인코더 클래스

    Args:
        pretrained_name (str, optional): 사전 학습된 모델 이름
        output_dim (int, optional): 출력 임베딩 차원

    Attributes:
        tokenizer (AutoTokenizer): BERT 토크나이저
        bert (AutoModel): 사전 학습된 DistilBERT 모델
        linear (nn.Linear): BERT 출력을 output_dim 크기로 변환하는 선형 레이어
    '''

    def __init__(self, pretrained_name="distilbert-base-uncased", output_dim=64):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.linear = torch.nn.Linear(768, output_dim)
    
    def forward(self, texts):
        '''
        입력된 텍스트를 DistilBERT를 통해 임베딩 벡터로 변환.
        
        '''

        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v for k, v in inputs.items()}

        outputs = self.bert(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :] 
        x = self.linear(cls_emb) 
        x = F.relu(x)
        return x


class GenreClusterEncoder(nn.Module):
    '''
    장르 클러스터 정보를 활용한 장르 임베딩 인코더

    Args:
        clusters_dict (dict): 장르 클러스터 정보 딕셔너리
        cluster_embeds (dict): 클러스터별 사전 학습된 임베딩
        config (OmegaConf): 모델 설정 정보

    Attributes:
        clusters_dict (dict): 클러스터별 장르 매핑
        cluster_embeds (dict): 클러스터별 임베딩 벡터
        tokenizer (AutoTokenizer): BERT 토크나이저
        bert (AutoModel): 사전 학습된 BERT 모델
        linear (nn.Linear): 장르 임베딩을 변환하는 선형 레이어
    '''
    def __init__(self, clusters_dict, cluster_embeds, config):

        super().__init__()
        self.clusters_dict = clusters_dict
        self.cluster_embeds = {cid: emb.clone().detach() for cid, emb in cluster_embeds.items()}  # No grad
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.bert_pretrained)
        self.bert = AutoModel.from_pretrained(config.model.bert_pretrained)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(768, config.model.output_dim) 

    def infer_cluster_for_genres(self, genres, device):
        '''        
        입력된 장르 리스트에 대한 클러스터 임베딩 생성
        
        '''

        known_cluster_embs = []
        unknown_genres = []

        # Check if genres belong to existing clusters
        for g in genres:
            found = False
            for cid, glist in self.clusters_dict.items():
                if g in glist:
                    known_cluster_embs.append(self.cluster_embeds[cid].to(device))
                    found = True
                    break
            if not found:
                unknown_genres.append(g)

        # Handle unknown genres using BERT
        if unknown_genres:
            inputs = self.tokenizer(unknown_genres, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.bert(**inputs)
            texts_emb = outputs.last_hidden_state[:, 0, :] 
            unknown_emb = self.linear(texts_emb).mean(dim=0) 
            known_cluster_embs.append(F.relu(unknown_emb))

        # Return zero vector if no known embeddings exist
        if len(known_cluster_embs) == 0:
            return torch.zeros(self.linear.out_features, device=device)
        else:
            return torch.stack(known_cluster_embs, dim=0).mean(dim=0)

    def forward(self, genres_batch):
        '''
        장르 배치 데이터를 입력받아 임베딩 벡터를 반환
        '''
        device = next(self.parameters()).device
        batch_embeddings = [self.infer_cluster_for_genres(genres, device) for genres in genres_batch]
        return torch.stack(batch_embeddings, dim=0)
