import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class BERTTextEncoder(nn.Module):
    """
    Freeze DistilBERT
    -> use only for forward calculation(not updated) 
    """
    def __init__(self, pretrained_name="distilbert-base-uncased", output_dim=64):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(768, output_dim)
    
    def forward(self, text):
        text = str(text)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.bert(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰 임베딩
        x = self.linear(cls_emb)
        x = F.relu(x)
        return x.squeeze(0)  # (output_dim,)

class ArtistEncoder(nn.Module):
    """
    Encode multiple artists
    ex) ["Taylor Swift", "Harry Styles", ...]
    2) LabelEncoding each artist -> nn.Embedding
    3) Mean Pooling and return (embed_dim,)
    """
    def __init__(self, artist_list, embed_dim=64):
        super().__init__()
        self.artist2id = {artist: i for i, artist in enumerate(artist_list)}
        self.embedding = nn.Embedding(len(self.artist2id), embed_dim)

    def forward(self, artists):
        if isinstance(artists, str):
            artists = [artists]
        if not isinstance(artists, list):
            raise ValueError("artist 값은 문자열 리스트 타입입니다.")
        
        indices = [self.artist2id.get(a, 0) for a in artists]
        if len(indices) == 0:
            indices = [0]
        
        idx_tensor = torch.LongTensor(indices)
        emb = self.embedding(idx_tensor)
        emb_mean = emb.mean(dim=0)
        emb_out = F.relu(emb_mean)
        return emb_out  # (embed_dim,)

class NumericEncoder(nn.Module):
    """
    Linear Layer: Embedding numeric values
    """
    def __init__(self, input_dim=2, output_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, output_dim)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.relu(x)
        return x.squeeze(0)  # (output_dim,)

class SongEncoder(nn.Module):
    """
    (artist, track, playlist, genres, listeners, length)
    -> MHA -> FC(32) -> L2 norm
    """
    def __init__(self,
                 artist_list,
                 bert_pretrained="distilbert-base-uncased",
                 mha_embed_dim=64,
                 mha_heads=4,
                 final_dim=32
                 ): 
        super().__init__()
        self.artist_encoder   = ArtistEncoder(artist_list, embed_dim=mha_embed_dim)
        self.track_encoder    = BERTTextEncoder(pretrained_name=bert_pretrained, output_dim=mha_embed_dim)
        self.playlist_encoder = BERTTextEncoder(pretrained_name=bert_pretrained, output_dim=mha_embed_dim)
        self.genres_encoder   = BERTTextEncoder(pretrained_name=bert_pretrained, output_dim=mha_embed_dim)
        self.numeric_encoder  = NumericEncoder(input_dim=2, output_dim=mha_embed_dim)

        self.mha = nn.MultiheadAttention(embed_dim=mha_embed_dim, num_heads=mha_heads, batch_first=True)
        self.final_fc = nn.Linear(mha_embed_dim, final_dim)

        nn.init.xavier_uniform_(self.final_fc.weight)

    def forward(self, artist, track, playlist, listeners, length, genres):
        artist_emb   = self.artist_encoder(artist)  
        track_emb    = self.track_encoder(track)    
        playlist_emb = self.playlist_encoder(playlist)
        genres_emb   = self.genres_encoder(genres)

        num_tensor = torch.tensor([listeners, length], dtype=torch.float)
        numeric_emb = self.numeric_encoder(num_tensor)

        features = torch.stack([artist_emb, track_emb, playlist_emb, genres_emb, numeric_emb], dim=0).unsqueeze(0)
        
        attn_output, _ = self.mha(features, features, features)
        agg_vector = attn_output.mean(dim=1)

        final_emb = self.final_fc(agg_vector)
        final_emb = F.normalize(final_emb, p=2, dim=1)
        return final_emb.squeeze(0)  # (final_dim,)

class GenreEncoder(nn.Module):
    """
    Genre(str list) -> DistilBERT(Freeze) -> (32) -> L2 norm
    """
    def __init__(self, pretrained_name="distilbert-base-uncased", embed_dim=32):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.linear = nn.Linear(768, embed_dim)

        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, genres):
        if isinstance(genres, list):
            text = " ".join(genres)
        else:
            text = str(genres)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.bert(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰 임베딩

        x = self.linear(cls_emb)
        x = F.relu(x)
        x = F.normalize(x, p=2, dim=1)
        return x.squeeze(0)  # (embed_dim,)
