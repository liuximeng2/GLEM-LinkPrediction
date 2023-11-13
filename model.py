import torch
import torch.nn as nn
import torch.nn.functional as F
from gae.layers import GraphConvolution
from transformers import BertModel
from transformers.modeling_outputs import TokenClassifierOutput

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, hidden_dim2, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, hidden_dim2)
        self.dropout = dropout
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return self.dc(x), x

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class BertClassifier(nn.Module):
    def __init__(self, model, dropout = 0):
        super(BertClassifier, self).__init__()
        self.bert = model
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.dropout(outputs['hidden_states'][-1])
        cls_token_emb = emb.permute(1, 0, 2)[0]
        print(f"cls_tokem size: {cls_token_emb.size()}")
        logits = self.dc(cls_token_emb)
        print(f"logit size: {logits.size()}")
        #loss = F.binary_cross_entropy_with_logits(logits, labels)
        #return TokenClassifierOutput(loss=loss, logits=logits)
        return logits

