"""CASMI paso 2 — paso2_lib.py
Utilidades compartidas: carga del pool de huellas, Tanimoto continuo,
recuperación por Tanimoto contra el pool completo y métricas.

Tanimoto continuo: para a (predicha, continua, clip [0,1]) y b (binaria),
T = a·b / (|a|^2 + |b|^2 - a·b).
"""
import time, resource
import numpy as np
from scipy import sparse

T0 = time.time()
def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time()-T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)

NBITS = 2048


def load_pool(inter):
    """Pool de huellas: CSR float32 (275,810 x 2048) + nº de bits por molécula."""
    X = sparse.load_npz(inter / "huellas_pool.npz").tocsr()
    Xf = X.astype(np.float32)
    nbits = np.diff(X.indptr).astype(np.float32)
    log(f"pool huellas: {X.shape[0]:,} x {X.shape[1]:,}, nnz {X.nnz:,}, "
        f"bits/mol media {nbits[nbits>0].mean():.1f}")
    del X
    return Xf, nbits


def tanimoto_rows(P, Y):
    """Tanimoto continuo por fila entre P (n x 2048 float32, clip [0,1])
    e Y (n x 2048 CSR bool). Devuelve array (n,) float32. También la
    versión binarizada (top-100 bits de P por fila)."""
    P = np.clip(np.asarray(P, dtype=np.float32), 0.0, 1.0)
    p2 = np.einsum("ij,ij->i", P, P)
    y2 = np.diff(Y.indptr).astype(np.float32)
    # dot por fila: P * Y en los nnz de Y
    Yc = Y.tocsr()
    dots = np.asarray(Yc.multiply(P).sum(axis=1)).ravel().astype(np.float32)
    den = p2 + y2 - dots
    t = np.zeros(len(P), dtype=np.float32)
    m = den > 0
    t[m] = dots[m] / den[m]
    # binarizada top-100
    k = 100
    Pb = np.zeros_like(P)
    idx = np.argpartition(-P, k, axis=1)[:, :k]
    np.put_along_axis(Pb, idx, 1.0, axis=1)
    pb2 = np.einsum("ij,ij->i", Pb, Pb)
    dotsb = np.asarray(Yc.multiply(Pb).sum(axis=1)).ravel().astype(np.float32)
    denb = pb2 + y2 - dotsb
    tb = np.zeros(len(P), dtype=np.float32)
    mb = denb > 0
    tb[mb] = dotsb[mb] / denb[mb]
    return t, tb


def retrieval(P, pool, pool_nbits, topk=25, batch=256, true_codes=None):
    """Ordena el pool por Tanimoto continuo contra cada fila de P.
    Devuelve (top_idx (n, topk) int32, top_sim (n, topk) float32,
    ranks (n,) int64 con el rango 1-based del true_code en el top-25,
    0 si ausente; None si true_codes es None)."""
    P = np.clip(np.asarray(P, dtype=np.float32), 0.0, 1.0)
    n = P.shape[0]
    N = pool.shape[0]
    top_idx = np.zeros((n, topk), dtype=np.int32)
    top_sim = np.zeros((n, topk), dtype=np.float32)
    ranks = np.zeros(n, dtype=np.int64) if true_codes is not None else None
    p2 = np.einsum("ij,ij->i", P, P)
    for s in range(0, n, batch):
        e = min(n, s + batch)
        Pb = P[s:e]
        D = Pb @ pool.T            # (b, N) float32
        den = pool_nbits[None, :].astype(np.float32) + p2[s:e][:, None]
        den -= D
        np.divide(D, den, out=D, where=den > 0)
        D[den <= 0] = 0.0
        del den
        for i in range(e - s):
            row = D[i]
            if topk >= N:
                part = np.argsort(-row)
            else:
                part = np.argpartition(-row, topk)[:topk]
                part = part[np.argsort(-row[part])]
            top_idx[s + i] = part.astype(np.int32)
            top_sim[s + i] = row[part]
            if true_codes is not None:
                hit = np.where(part == true_codes[s + i])[0]
                ranks[s + i] = int(hit[0] + 1) if len(hit) else 0
        del D
        if (s // batch) % 10 == 0:
            log(f"retrieval: {e:,}/{n:,} consultas")
    log(f"retrieval completo: {n:,} consultas x pool {N:,}")
    return top_idx, top_sim, ranks
