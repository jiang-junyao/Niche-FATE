"""Graph convolutional autoencoder for the neighbourhood (niche) embedding.
Two mean-aggregation GCN layers with tied encoder/decoder weights on a k-NN spatial graph; all sections are trained
jointly (block-diagonal graph), so embeddings are comparable across time points."""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def gcn_embed(Z_list, A_list, hid=128, out=30, epochs=500, lr=1e-3, seed=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    A = sp.block_diag(A_list).tocoo()
    At = torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data.astype(np.float32), A.shape, device=device).coalesce()
    X = torch.tensor(np.vstack(Z_list), dtype=torch.float32, device=device)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    W1 = nn.Parameter(torch.randn(X.shape[1], hid, device=device) * 0.05)
    W2 = nn.Parameter(torch.randn(hid, out, device=device) * 0.05)
    mm = lambda M: torch.sparse.mm(At, M)
    enc = lambda x: mm(torch.nn.functional.elu(mm(x @ W1)) @ W2)
    dec = lambda h: mm(torch.nn.functional.elu(mm(h @ W2.T)) @ W1.T)
    opt = torch.optim.Adam([W1, W2], lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        loss = ((dec(enc(X)) - X) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        H = enc(X).cpu().numpy()
    H = (H - H.mean(0)) / (H.std(0) + 1e-6)
    out_, o = [], 0
    for Z in Z_list:
        out_.append(H[o:o + len(Z)].astype(np.float32)); o += len(Z)
    return out_
