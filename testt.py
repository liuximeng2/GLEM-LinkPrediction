import torch
import argparse
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
import time
import numpy as np
import scipy.sparse as sp
from torch import optim
import torch.nn.functional as F
from gae.model import BertClassifier
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, load_rawtext_data
from transformers.models.auto import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers.trainer import Trainer, TrainingArguments, IntervalStrategy
import argparse
import torch_geometric
#from ogb.nodeproppred import Evaluator
import os

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, raw_texts, pyg_data, labels=None):
        self.encodings = encodings
        self.labels = labels
        self.raw_texts = raw_texts
        self.data_obj = pyg_data

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx].flatten(),
            'attention_mask': self.encodings['attention_mask'][idx].flatten(),
        }
        item['node_id'] = idx
        if self.labels != None:
            item["labels"] = self.labels[idx].to(torch.long)
        return item

    def __len__(self):
        return len(self.raw_texts)

def parse_args():
    parser = argparse.ArgumentParser(description='LM training')
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./lmoutput")
    parser.add_argument('--checkpoint_dir', type=str, default="./lmcheckpoint")

    args = parser.parse_args()
    return args

args = parse_args()
data_obj = torch.load('gae/data/cora_random_sber_embbed.pt', map_location='cpu')
seed = args.seed
n_labels = data_obj.y.max().item() + 1
data_obj.train_mask = data_obj.train_masks[seed]
data_obj.val_mask = data_obj.val_masks[seed]
data_obj.test_mask = data_obj.test_masks[seed]
train_steps = data_obj.x.shape[0] // (32 + 1)
eval_steps = 50000 // 32
warmup_step = int(0.6 * train_steps)

training_args = TrainingArguments(
            output_dir=args.output_dir,
            do_train=True,
            do_eval=True,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=eval_steps,
            save_steps=eval_steps,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            load_best_model_at_end=True,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            warmup_steps=warmup_step,
            num_train_epochs=args.epochs,
            dataloader_num_workers=1,
            fp16=True,
            dataloader_drop_last=True,
            local_rank=0,
            report_to='none'
        )
print("downloading pretrained model")
pretrained_model = AutoModel.from_pretrained("microsoft/deberta-base")
model = BertClassifier(pretrained_model)
print("downloading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
X = tokenizer(data_obj.raw_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

adj,_ = load_rawtext_data('cora')
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = torch.FloatTensor(adj_label.toarray())

optimizer = optim.Adam(model.parameters(), lr=0.01)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
X = X.to(device)
adj_label = adj_label.to(device)


os._exit(0)

for epoch in range(5):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    # Forward pass
    print("Training")
    recovered = model(X["input_ids"], X["attention_mask"])
    print("Calculating Loss")
    loss = F.binary_cross_entropy_with_logits(recovered, adj_label)
    loss.backward()
    cur_loss = loss.item()
    optimizer.step()

    hidden_emb = recovered.cpu()
    hidden_emb = hidden_emb.data.numpy()

    roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
    #if (epoch + 1) % 10 == 0:
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t))

print("Training Finished")
