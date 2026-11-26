"""CASMI baseline 1 — casmi_lib.py
Funciones compartidas: carga de librería ordenada por precursor, similitudes
coseno y de entropía espectral, agregación por molécula y MRR@25.

Bineado: bins de 0.01 Da (0-2000 Da), top-100 picos por intensidad, intensidades
crudas bineadas (la CSR guarda intensidades crudas; coseno y entropía se
derivan de ella).
- Coseno: intensidades con raíz cuadrada y normalización L2 (estándar en
  búsqueda en librerías, p. ej. GNPS/matchms).
- Entropía espectral (Li et al., Anal. Chem. 2021): con p = intensidades
  normalizadas L1, S = -sum(p ln p); sim = 1 - (2*S_AB - S_A - S_B)/ln(4),
  donde S_AB es la entropía del espectro fusionado. Se calcula solo con el
  término de intersección: S_AB = ln2 + (S_A+S_B)/2 - (1/2)*sum_bins_comunes
  [g(a+b)-g(a)-g(b)], g(x)=x ln x. Verificado: idénticos -> 1, disjuntos -> 0.
"""
import time, resource
import numpy as np
from scipy import sparse

T0 = time.time()
def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time()-T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)

NBINS = 200_000
LN2 = np.log(2.0)
LN4 = np.log(4.0)


def g(x):
    """x ln x, con g(0)=0."""
    out = np.zeros_like(x, dtype=np.float64)
    m = x > 0
    out[m] = x[m] * np.log(x[m])
    return out


def load_library(inter, tag):
    """Carga CSR + meta y ordena filas por precursor_mz (para searchsorted)."""
    X = sparse.load_npz(inter / f"{tag}.npz").tocsr()
    meta = np.load(inter / f"{tag}_meta.npz", allow_pickle=True)
    prec = meta["precursor_mz"].astype(np.float64)
    order = np.argsort(prec, kind="stable")
    X = X[order]
    log(f"librería {tag}: {X.shape[0]:,} filas, {X.nnz:,} nnz, prec {prec.min():.1f}-{prec.max():.1f}")
    return X, prec[order], meta["mol"][order], meta["ingest_lib"][order]


def load_queries(inter, tag):
    X = sparse.load_npz(inter / f"{tag}.npz").tocsr()
    meta = np.load(inter / f"{tag}_meta.npz", allow_pickle=True)
    log(f"queries {tag}: {X.shape[0]:,} filas, {X.nnz:,} nnz")
    return X, meta["precursor_mz"].astype(np.float64), meta["mol"], meta["ingest_lib"]


def row_stats(X):
    """Por fila: suma cruda (para normas) y entropía S de p=L1-normalizado."""
    rowsum = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    rowsum[rowsum <= 0] = 1.0
    # entropía: -sum(p ln p), p = data/rowsum[row]
    nrows = X.shape[0]
    S = np.zeros(nrows, dtype=np.float64)
    indptr = X.indptr
    data = X.data.astype(np.float64)
    # por bloques de filas para acotar memoria
    B = 200_000
    for s in range(0, nrows, B):
        e = min(nrows, s + B)
        sl = slice(indptr[s], indptr[e])
        d = data[sl]
        rep = np.repeat(rowsum[s:e], np.diff(indptr[s:e + 1]))
        p = d / rep
        S[s:e] = -np.bincount(np.repeat(np.arange(e - s), np.diff(indptr[s:e + 1])),
                              weights=np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0),
                              minlength=e - s)
    norm_cos = np.sqrt(rowsum)   # ||sqrt(v)||_2 = sqrt(sum v)
    return rowsum, S, norm_cos


def merge_bins(X, k):
    """Fusiona bins adyacentes: nuevo bin = viejo // k (p. ej. k=5: 0.01 -> 0.05 Da)."""
    coo = X.tocoo()
    coo.col //= k
    Y = coo.tocsr()
    Y.sum_duplicates()
    return Y


def candidate_range(prec_sorted, pmz, modo):
    """Ventana de precursor. modo 'ppm': max(10 ppm, 0.002 Da); 'da': +-0.01 Da."""
    if modo == "ppm":
        tol = max(pmz * 10e-6, 0.002)
    elif modo == "da":
        tol = 0.01
    else:
        raise ValueError(modo)
    lo = int(np.searchsorted(prec_sorted, pmz - tol))
    hi = int(np.searchsorted(prec_sorted, pmz + tol))
    return lo, hi


def score_query(X, prec_sorted, mol_sorted, rowsum, S_row, norm_cos, q_row, q_pmz, modo):
    """Coseno y entropía de un espectro consulta contra sus candidatos.
    Devuelve (códigos mol candidatos, cos, ent) o (None, None, None) si no hay candidatos."""
    lo, hi = candidate_range(prec_sorted, q_pmz, modo)
    if hi <= lo:
        return None, None, None
    C = X[lo:hi]
    nc = hi - lo
    # --- consulta: vectores derivados ---
    q_data = q_row.data.astype(np.float64)
    q_ind = q_row.indices
    q_sum = q_data.sum()
    if q_sum <= 0:
        return None, None, None
    q_sqrt = np.sqrt(q_data)
    q_norm_cos = np.sqrt(q_sum)
    q_l1 = q_data / q_sum
    S_q = -float(np.sum(np.where(q_l1 > 0, q_l1 * np.log(np.where(q_l1 > 0, q_l1, 1.0)), 0.0)))
    q_dense = np.zeros(X.shape[1], dtype=np.float64)
    q_dense[q_ind] = q_sqrt
    # --- coseno: sqrt(C) @ q_sqrt / normas ---
    Csqrt = C.copy()
    Csqrt.data = np.sqrt(C.data.astype(np.float64)).astype(np.float32)
    cos = np.asarray(Csqrt.dot(q_dense)).ravel()
    cos = cos / (norm_cos[lo:hi] * q_norm_cos)
    np.clip(cos, -1.0, 1.0, out=cos)
    # --- entropía: valores de candidatos en los bins de la consulta ---
    Cq = C[:, q_ind].toarray().astype(np.float64)     # (nc, n_bins_q)
    rs = rowsum[lo:hi][:, None]
    pc = Cq / rs                                       # L1-normalizado por candidato
    s = q_l1[None, :] + pc
    term = np.sum(g(s) - g(q_l1)[None, :] - g(pc), axis=1)
    S_AB = LN2 + (S_q + S_row[lo:hi]) / 2.0 - term / 2.0
    ent = 1.0 - (2.0 * S_AB - S_q - S_row[lo:hi]) / LN4
    np.clip(ent, 0.0, 1.0, out=ent)
    return mol_sorted[lo:hi], cos, ent


def rank_molecule(scores_por_espectro, mol_cand_por_espectro):
    """Agrega por molécula: max score por inchikey14 (código) en todos sus espectros.
    Devuelve lista ordenada de códigos (mejor primero)."""
    best = {}
    for mol_cand, sc in zip(mol_cand_por_espectro, scores_por_espectro):
        for m, s in zip(mol_cand, sc):
            if s > best.get(m, -1.0):
                best[m] = float(s)
    return sorted(best, key=lambda m: -best[m])


def mrr_from_ranks(ranks):
    """ranks: lista de rangos 1-based (0 = ausente de top-25). Devuelve dict de métricas."""
    r = np.asarray(ranks, dtype=np.float64)
    rr = np.where(r > 0, 1.0 / np.where(r > 0, r, 1.0), 0.0)
    return {"n": int(len(r)), "mrr25": float(rr.mean()),
            "top1": float((r == 1).mean()), "top5": float(((r >= 1) & (r <= 5)).mean()),
            "recall25": float((r > 0).mean()),
            "rank_mediana_presentes": float(np.median(r[r > 0])) if (r > 0).any() else None}
