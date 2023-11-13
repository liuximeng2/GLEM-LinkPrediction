
import numpy as np
from gae.utils import load_data, load_rawtext_data
import torch
import networkx as nx
import sys
from transformers import DebertaTokenizer, DebertaModel
from sentence_transformers import SentenceTransformer


data = torch.load('gae/data/cora_random_sbert.pt', map_location='cpu')

data.raw_texts = data.raw_text
data.category_names = [data.label_names[i] for i in data.y.tolist()]
d_name = 'cora'
entity_pt = torch.load('gae/data/cora_entity.pt', map_location="cpu")
data.entity = entity_pt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

def sbert(device):
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device).to(device)
    return model

def get_sbert_embedding(texts):
    sbert_model = sbert('cuda')
    sbert_embeds = sbert_model.encode(texts, batch_size=8, show_progress_bar=True)
    return torch.tensor(sbert_embeds)

# Load the tokenizer and model
#tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
#encoded_input = tokenizer(data.raw_text[0], return_tensors='pt')
print("Getting embedding")
data.x = get_sbert_embedding(data.raw_text)

torch.save(data, "gae/data/cora_random_sbert_embedded.pt")
print("Saved")