from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn.functional as F

from model import*
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, load_rawtext_data

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')

args = parser.parse_args()


def gae_for(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    print("Using {} dataset".format(args.dataset_str))
    adj, features = load_rawtext_data(args.dataset_str)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()]).to(device)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    if args.model == 'gcn_vae':
        model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    elif args.model == 'GCN':
        model = GCN(feat_dim, args.hidden1, args.hidden2, args.dropout)
    elif args.model == 'LM':
        model = LMDecoder()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.to(device)
    features = features.to(device)
    adj_norm = adj_norm.to(device)
    adj_label = adj_label.to(device)

    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        if args.model == 'gcn_vae':
            print("Using GVAE")
            recovered, mu, logvar = model(features, adj_norm)
            loss = loss_function(preds=recovered, labels=adj_label,
                                 mu=mu, logvar=logvar, n_nodes=n_nodes,
                                 norm=norm, pos_weight=pos_weight)
        elif args.model == 'GCN':
            print("Using GCN")
            recovered, mu = model(features, adj_norm)
            loss = norm * F.binary_cross_entropy_with_logits(recovered, adj_label, pos_weight=pos_weight)

        elif args.model == 'LM':
            print("Using LM")
            adj = model()

        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        cpu_mu = mu.cpu()
        hidden_emb = cpu_mu.data.numpy()

        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
        if (epoch + 1) % 10 == 0:

            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t)
                  )

    print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))

if __name__ == '__main__':
    gae_for(args)
