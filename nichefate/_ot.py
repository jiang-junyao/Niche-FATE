"""Transport solver (internal).: ONE model, the FGW weight alpha is a parameter (alpha = 0 -> plain unbalanced OT).
Linear cost   M_ij = |F1_i - F2_j|^2          (F = concatenated, median-normalised expression / niche / signal features)
Quadratic     D1, D2 = squared Euclidean distances within each slice (coordinates scaled so median(D) = 1)
Objective     (1 - alpha) <M, P> + alpha * GW_square_loss(D1, D2, P) + eps * KL, marginals KL-relaxed with tau.
Solved by entropic projected gradient (as POT / moscot): each outer step solves an unbalanced Sinkhorn on
C = (1 - alpha) M + alpha * grad_GW(P), warm-started. Because D1, D2 are squared Euclidean (rank-4 factorisation, as in moscot
Suppl. Note 2), grad_GW never needs n x n matrices, and every cost is of the form a_i + b_j - 2 G1_i.G2_j, so nothing of size
n x m is ever stored: memory O((n + m) d), any slice size."""
import numpy as np, torch
def _lse(G1, G2, a, b, h, eps, chunk):
    out = torch.empty(G1.shape[0], dtype=G1.dtype, device=G1.device)
    for s in range(0, G1.shape[0], chunk):
        C = a[s:s + chunk, None] + b[None] - 2 * G1[s:s + chunk] @ G2.T
        out[s:s + chunk] = torch.logsumexp((h[None] - C) / eps, 1)
    return out
def _sinkhorn(G1, G2, a, b, eps, tau, f, g, n_iter, tol, chunk, la=None, lb=None, tau_b=None):
    n, m = len(G1), len(G2); la = -np.log(n) if la is None else la; lb = -np.log(m) if lb is None else lb; tb = tau if tau_b is None else tau_b
    for it in range(n_iter):
        fn = tau * eps * (la - _lse(G1, G2, a, b, g, eps, chunk)); gn = tb * eps * (lb - _lse(G2, G1, b, a, fn, eps, chunk))
        err = (fn - f).abs().max().item() + (gn - g).abs().max().item(); f, g = fn, gn
        if err < tol: break
    return f, g, it + 1
def _stats(G1, G2, a, b, f, g, eps, B1, B2, chunk, lab2=None):
    """row sums p, column sums q, K = B1^T P B2, optional per-row mass on each target label"""
    n, m = len(G1), len(G2); p = torch.zeros(n, dtype=torch.float64, device=G1.device); q = torch.zeros(m, dtype=torch.float64, device=G1.device)
    K = torch.zeros(B1.shape[1], B2.shape[1], dtype=torch.float64, device=G1.device) if B1 is not None else None; R = None
    if lab2 is not None: R = torch.zeros(n, lab2.shape[1], dtype=torch.float64, device=G1.device)
    for s in range(0, n, chunk):
        C = a[s:s + chunk, None] + b[None] - 2 * G1[s:s + chunk] @ G2.T
        P = torch.exp(((f[s:s + chunk, None] + g[None] - C) / eps).double())
        p[s:s + chunk] = P.sum(1); q += P.sum(0)
        if B1 is not None: K += B1[s:s + chunk].double().T @ (P @ B2.double())
        if R is not None: R[s:s + chunk] = P @ lab2
    return p, q, K, R
def _fact(X):   # squared Euclidean distance matrix D = A B^T
    s = (X ** 2).sum(1, keepdim=True); one = torch.ones_like(s); return torch.cat([s, one, -2 * X], 1), torch.cat([one, s, X], 1)
def solve(F1, F2, X1, X2, alpha, eps, tau, lab1, lab2, n_outer=50, outer_tol=1e-4, n_iter=2000, tol=1e-6, chunk=2048, device=None, dtype=torch.float32, verbose=False, inner_tol=None, return_cell=False, w1=None, tau_b=None):
    """w1: optional source marginal weights (e.g. moscot-style prior growth), normalised to sum 1; tau = tau_a, tau_b defaults to tau."""
    """Returns dict(T, u1, u2, top, rowmass) (type-level summary) + diagnostics. X1, X2: coordinates already scaled."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    T_ = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    F1, F2 = T_(F1), T_(F2); n, m = len(F1), len(F2); x2, y2 = (F1 ** 2).sum(1), (F2 ** 2).sum(1)
    u1, i1 = np.unique(lab1, return_inverse=True); u2, i2 = np.unique(lab2, return_inverse=True)
    O2 = torch.zeros(m, len(u2), dtype=torch.float64, device=device); O2[torch.arange(m), torch.as_tensor(i2, device=device)] = 1
    f = torch.zeros(n, dtype=dtype, device=device); g = torch.zeros(m, dtype=dtype, device=device); its = []
    LA = None if w1 is None else torch.as_tensor(np.log(np.asarray(w1, float) / np.sum(w1)), dtype=dtype, device=device)
    SK = lambda G1_, G2_, a_, b_, f_, g_, tl: _sinkhorn(G1_, G2_, a_, b_, eps, tau, f_, g_, n_iter, tl, chunk, la=LA, tau_b=tau_b)
    if alpha == 0:
        f, g, it = SK(F1, F2, x2, y2, f, g, tol); its.append(it); G1, G2, a, b = F1, F2, x2, y2
    else:
        A1, B1 = _fact(T_(X1)); A2, B2 = _fact(T_(X2))
        p = (torch.full((n,), 1.0 / n, dtype=torch.float64, device=device) if w1 is None else torch.as_tensor(np.asarray(w1, float) / np.sum(w1), dtype=torch.float64, device=device)); q = torch.full((m,), 1.0 / m, dtype=torch.float64, device=device)
        K = torch.outer(B1.double().T @ p, B2.double().T @ q)
        for t in range(n_outer):
            S1 = B1.double().T @ (B1.double() * p[:, None]); S2 = B2.double().T @ (B2.double() * q[:, None])
            c1 = ((A1.double() @ S1) * A1.double()).sum(1); c2 = ((A2.double() @ S2) * A2.double()).sum(1)      # sum_k D1_ik^2 p_k, sum_l D2_jl^2 q_l
            # grad_GW = 2 * (c1_i + c2_j - 2 (D1 P D2)_ij),  (D1 P D2)_ij = A1_i K A2_j^T  with K = B1^T P B2
            a = ((1 - alpha) * x2.double() + 2 * alpha * c1).to(dtype); b = ((1 - alpha) * y2.double() + 2 * alpha * c2).to(dtype)
            G1 = torch.cat([(1 - alpha) * F1, (2 * alpha * (A1.double() @ K)).to(dtype)], 1); G2 = torch.cat([F2, A2], 1)
            f, g, it = SK(G1, G2, a, b, f, g, inner_tol or tol); its.append(it)
            p, q, Kn, _ = _stats(G1, G2, a, b, f, g, eps, B1, B2, chunk)
            d = ((Kn - K).abs().max() / K.abs().max()).item(); K = Kn
            if verbose: print("outer", t, "inner", it, "dK", d, flush=True)
            if d < outer_tol: break
        if inner_tol:                                  # final polish of the inner problem at full tolerance
            f, g, it = SK(G1, G2, a, b, f, g, tol); its.append(it)
    _, _, _, R = _stats(G1, G2, a, b, f, g, eps, None, None, chunk, lab2=O2)
    R = R.cpu().numpy(); O1 = np.eye(len(u1))[i1]
    out = dict(T=O1.T @ R, u1=u1, u2=u2, top=u2[R.argmax(1)], rowmass=R.sum(1), n_inner=np.array(its))
    if return_cell: out["cell_mass"] = R.astype(np.float32)          # per source cell: transported mass to each target label (u2)
    return out
