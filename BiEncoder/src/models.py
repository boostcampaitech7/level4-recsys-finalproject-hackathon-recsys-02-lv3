import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class BERTTextEncoder(nn.Module):
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
    def __init__(self, config, playlist_info=None, cluster_embeds=None, clusters_dict=None): 
        super().__init__()
        self.artist_encoder = PlaylistAwareArtistEncoder(playlist_info, output_dim=config.model.mha_embed_dim)
        self.track_encoder = BERTTextEncoder(config)
        self.playlist_encoder = BERTTextEncoder(config)

        self.genres_encoder = GenreClusterEncoder(clusters_dict, cluster_embeds, config)

        self.numeric_encoder = NumericEncoder(config)

        self.mha = nn.MultiheadAttention(embed_dim=config.model.mha_embed_dim, num_heads=config.model.mha_heads, batch_first=True)
        self.final_fc = nn.Linear(config.model.mha_embed_dim, config.model.final_dim)
        nn.init.xavier_uniform_(self.final_fc.weight)

    def forward(self, artists, tracks, playlists, listeners, lengths, genres):
            
        artist_emb = self.artist_encoder(artists)
        track_emb = self.track_encoder(tracks)
        playlist_emb = self.playlist_encoder(playlists)
        genres_emb = self.genres_encoder(genres)

        num_tensor = torch.tensor(list(zip(listeners, lengths)), dtype=torch.float32).to(artist_emb.device)
        numeric_emb = self.numeric_encoder(num_tensor)

        features = torch.stack([artist_emb, track_emb, playlist_emb, genres_emb, numeric_emb], dim=1)
        
        attn_output, _ = self.mha(features, features, features)
        agg_vector = attn_output.mean(dim=1)

        final_emb = self.final_fc(agg_vector)

        return F.normalize(final_emb, p=2, dim=1)

class GenreEncoder(nn.Module):
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

class PlaylistAwareArtistEncoder(nn.Module):
    def __init__(self, playlist_info, output_dim=64):
        super().__init__()
        self.output_dim = output_dim
        self.artist2id = {}
        self.id2artist = {}
        self.playlist_info = playlist_info
        self.graph = self.create_artist_graph(playlist_info)
        
        # Dynamic artist registration
        unique_artists = list(self.graph.nodes())
        self.artist2id = {artist: idx for idx, artist in enumerate(unique_artists)}
        self.id2artist = {idx: artist for artist, idx in self.artist2id.items()} 
        
        self.embedding = nn.Embedding(len(unique_artists) + 100, output_dim)
        self.init_embeddings_with_graph_info()

    def create_artist_graph(self, playlist_info):
        G = nx.Graph()
        for playlist_artists in playlist_info.values():
            for i in range(len(playlist_artists)):
                for j in range(i+1, len(playlist_artists)):
                    G.add_edge(playlist_artists[i], playlist_artists[j])
        return G

    def register_new_artist(self, artist):
        if artist not in self.artist2id:
            new_id = len(self.artist2id)
            self.artist2id[artist] = new_id
            self.id2artist[new_id] = artist
            
            # Expand embedding layer
            if new_id >= self.embedding.num_embeddings:
                device = self.embedding.weight.device 
                new_embedding = nn.Embedding(new_id + 100, self.output_dim).to(device)
                new_embedding.weight.data[:self.embedding.num_embeddings] = self.embedding.weight.data.to(device)
                self.embedding = new_embedding

    def init_embeddings_with_graph_info(self):
        centrality = nx.pagerank(self.graph)
        for artist, idx in self.artist2id.items():
            self.embedding.weight.data[idx] = torch.randn(self.output_dim) * centrality.get(artist, 0.5)

    def forward(self, artists_batch):
        batch_embeddings = []
        device = self.embedding.weight.device

        for track_artists in artists_batch:
            # Dynamically register new artists
            for artist in track_artists:
                self.register_new_artist(artist)
            # Get embeddings for each artist in the track
            track_emb = [
                self.embedding(torch.tensor(self.artist2id[artist], device=device))
                for artist in track_artists
            ]
            # Average embeddings for the track
            batch_embeddings.append(torch.mean(torch.stack(track_emb), dim=0))
        
        return torch.stack(batch_embeddings)

class DistilBertTextEncoder(torch.nn.Module):
    def __init__(self, pretrained_name="distilbert-base-uncased", output_dim=64):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.linear = torch.nn.Linear(768, output_dim)
    
    def forward(self, texts):

        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v for k, v in inputs.items()}

        outputs = self.bert(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :] 
        x = self.linear(cls_emb) 
        x = F.relu(x)
        return x

class GenreClusterEncoder(nn.Module):
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
        known_cluster_embs = []
        unknown_genres = []

        for g in genres:
            found = False
            for cid, glist in self.clusters_dict.items():
                if g in glist:
                    known_cluster_embs.append(self.cluster_embeds[cid].to(device))
                    found = True
                    break
            if not found:
                unknown_genres.append(g)

        if unknown_genres:
            inputs = self.tokenizer(unknown_genres, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.bert(**inputs)
            texts_emb = outputs.last_hidden_state[:, 0, :] 
            unknown_emb = self.linear(texts_emb).mean(dim=0) 
            known_cluster_embs.append(F.relu(unknown_emb))

        if len(known_cluster_embs) == 0:
            return torch.zeros(self.linear.out_features, device=device)
        else:
            return torch.stack(known_cluster_embs, dim=0).mean(dim=0)

    def forward(self, genres_batch):
        device = next(self.parameters()).device
        batch_embeddings = [self.infer_cluster_for_genres(genres, device) for genres in genres_batch]
        return torch.stack(batch_embeddings, dim=0)
