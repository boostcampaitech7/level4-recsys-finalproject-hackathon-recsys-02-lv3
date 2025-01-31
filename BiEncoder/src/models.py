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
    def __init__(self, initial_artist_vocab_size=1000, bert_pretrained="distilbert-base-uncased", 
                 mha_embed_dim=64, mha_heads=4, final_dim=32, cluster_embeds=None, clusters_dict=None): 
        super().__init__()
        self.artist_encoder = DynamicArtistEncoder(initial_vocab_size=initial_artist_vocab_size, 
                                                 embed_dim=mha_embed_dim)
        self.track_encoder = BERTTextEncoder(pretrained_name=bert_pretrained, output_dim=mha_embed_dim)
        self.playlist_encoder = BERTTextEncoder(pretrained_name=bert_pretrained, output_dim=mha_embed_dim)

        self.genres_encoder = GenreClusterEncoder(clusters_dict, cluster_embeds, pretrained_name=bert_pretrained, embed_dim=mha_embed_dim)

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

class DynamicArtistEncoder(nn.Module):
    def __init__(self, initial_vocab_size=1000, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.artist2id = {"<UNK>": 0}
        self.id2artist = ["<UNK>"]
        self.embedding = nn.EmbeddingBag(
            num_embeddings=initial_vocab_size, 
            embedding_dim=embed_dim, 
            mode='mean'
        )
        
    def add_artist(self, artist):
        if artist not in self.artist2id:
            new_id = len(self.artist2id)
            if new_id >= self.embedding.num_embeddings:
                # new_embedding = nn.EmbeddingBag(new_id + 100, self.embed_dim, mode='mean')
                # with torch.no_grad():
                #     new_embedding.weight[:self.embedding.num_embeddings] = self.embedding.weight
                # self.embedding = new_embedding

                device = self.embedding.weight.device  # 기존 임베딩이 있는 디바이스 확인

                new_embedding = nn.EmbeddingBag(new_id + 100, self.embed_dim, mode='mean').to(device)  # ✅ GPU로 이동
                with torch.no_grad():
                    new_embedding.weight[:self.embedding.num_embeddings] = self.embedding.weight.to(device)  # ✅ GPU로 이동

                self.embedding = new_embedding

            
            self.artist2id[artist] = new_id
            self.id2artist.append(artist)
    
    def forward(self, artists_batch):
        flattened_indices = []
        offsets = [0]  
        total_length = 0

        for artists in artists_batch:
            for artist in artists:
                if artist not in self.artist2id:
                    self.add_artist(artist)
            
            indices = [self.artist2id[a] for a in artists] if len(artists) > 0 else [0]
            flattened_indices.extend(indices)  
            
            total_length += len(indices)
            offsets.append(total_length)

        offsets = offsets[:-1] 

        flattened_indices_t = torch.tensor(flattened_indices, dtype=torch.long, 
                                           device=self.embedding.weight.device)
        offsets_t = torch.tensor(offsets, dtype=torch.long, 
                                 device=self.embedding.weight.device)

        emb_out = self.embedding(flattened_indices_t, offsets_t)
        
        return F.relu(emb_out)

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
    def __init__(self, clusters_dict, cluster_embeds, pretrained_name="distilbert-base-uncased", embed_dim=64):

        super().__init__()
        self.clusters_dict = clusters_dict
        self.cluster_embeds = {cid: emb.clone().detach() for cid, emb in cluster_embeds.items()}  # No grad
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(768, embed_dim)

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
