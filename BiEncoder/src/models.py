import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class BERTTextEncoder(nn.Module):
    def __init__(self, pretrained_name="distilbert-base-uncased", output_dim=64):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(768, output_dim)
    
    def forward(self, texts):
        texts = [str(text) for text in texts]
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.linear.weight.device) for k, v in inputs.items()}
        
        outputs = self.bert(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        x = self.linear(cls_emb)
        x = F.relu(x)
        return x

class ArtistEncoder(nn.Module):
    def __init__(self, artist_list, embed_dim=64):
        super().__init__()
        self.artist2id = {artist: i for i, artist in enumerate(artist_list)}
        self.embedding = nn.Embedding(len(self.artist2id), embed_dim)

    def forward(self, artists_batch):
        if isinstance(artists_batch[0], str):
            artists_batch = [[a] for a in artists_batch]  # 단일 샘플도 배치 형태로 변환
        
        batch_indices = []
        for artists in artists_batch:
            indices = [self.artist2id.get(a, 0) for a in artists]
            if len(indices) == 0:
                indices = [0]
            batch_indices.append(indices)
        
        max_len = max(len(indices) for indices in batch_indices)
        padded_indices = [
            indices + [0] * (max_len - len(indices)) 
            for indices in batch_indices
        ]
        
        idx_tensor = torch.tensor(padded_indices, dtype=torch.long, device=self.embedding.weight.device)
        emb = self.embedding(idx_tensor)
        emb_mean = emb.mean(dim=1)
        return F.relu(emb_mean)


class NumericEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, output_dim)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.relu(x)

class SongEncoder(nn.Module):
    def __init__(self, artist_list, bert_pretrained="distilbert-base-uncased", 
                 mha_embed_dim=64, mha_heads=4, final_dim=32): 
        super().__init__()
        self.artist_encoder = ArtistEncoder(artist_list, embed_dim=mha_embed_dim)
        self.track_encoder = BERTTextEncoder(pretrained_name=bert_pretrained, output_dim=mha_embed_dim)
        self.playlist_encoder = BERTTextEncoder(pretrained_name=bert_pretrained, output_dim=mha_embed_dim)
        self.genres_encoder = BERTTextEncoder(pretrained_name=bert_pretrained, output_dim=mha_embed_dim)
        self.numeric_encoder = NumericEncoder(input_dim=2, output_dim=mha_embed_dim)

        self.mha = nn.MultiheadAttention(embed_dim=mha_embed_dim, num_heads=mha_heads, batch_first=True)
        self.final_fc = nn.Linear(mha_embed_dim, final_dim)
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
    def __init__(self, pretrained_name="distilbert-base-uncased", embed_dim=32):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.linear = nn.Linear(768, embed_dim)

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
