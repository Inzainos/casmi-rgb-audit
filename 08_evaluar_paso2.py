"""CASMI paso 2 — 08_evaluar_paso2.py
Evalúa una matriz de huellas predichas por molécula (val o spec):
  a) Tanimoto continuo y binarizado top-100 contra la huella real.
  b) Baseline nulo (huella media = frecuencias de bit) con la misma métrica.
  c) Recuperación: top-25 del pool por Tanimoto -> MRR@25 / Top-1 / Top-5 /
     Recall@25 de la molécula verdadera (mol-split: la molécula es nueva
     para el modelo pero su huella está en el pool, como en PubChem/COCONUT).
  d) Oráculo: la misma recuperación con la huella REAL como consulta
     (techo del retrieval dado el pool) + colisiones exactas de huella en
     el pool (moléculas distintas con la misma huella de Morgan: ni el
     oráculo puede distinguirlas).

Uso: python3 08_evaluar_paso2.py --pred intermedios/P_val_lineal.npy \
       --mol intermedios/P_val_mol_lineal.npy --tag lineal_val
"""
import argparse, time, json, sys
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).parent))
from casmi_lib import log, mrr_from_ranks
from paso2_lib import load_pool, tanimoto_rows, retrieval

BASE = Path("/home/hatch/workspace/snt-taxonomia/kaggle-casmi")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"


def hash_filas(X):
    """Hash 64-bit por fila (XOR de splitmix64 de los índices+1).
    Para detectar colisiones exactas de huella: mismo hash + mismo nº de
    bits -> verificación exacta comparando índices."""
    ind = X.indices.astype(np.uint64) + np.uint64(1)
    # splitmix64
    z = ind + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    h = np.bitwise_xor.reduceat(z, X.indptr[:-1])
    h[np.diff(X.indptr) == 0] = np.uint64(0)
    return h


def colisiones(pool_bool, h_pool, codigos):
    """Para cada código, nº de moléculas del pool con la huella EXACTA
    (mismo hash + mismo nº de bits + índices idénticos). Devuelve array."""
    nbits = np.diff(pool_bool.indptr)
    res = np.zeros(len(codigos), dtype=np.int64)
    # índice hash -> filas (solo hashes con duplicados)
    uniq_h, counts_h = np.unique(h_pool, return_counts=True)
    dup_h = set(uniq_h[counts_h > 1].tolist())
    from collections import defaultdict
    por_hash = defaultdict(list)
    for i in np.where(np.isin(h_pool, list(dup_h)))[0]:
        por_hash[int(h_pool[i])].append(int(i))
    for k, c in enumerate(codigos):
        h = int(h_pool[c])
        if h not in por_hash:
            res[k] = 0
            continue
        idx_c = pool_bool.indices[pool_bool.indptr[c]:pool_bool.indptr[c + 1]]
        n = 0
        for j in por_hash[h]:
            if j == c or nbits[j] != nbits[c]:
                continue
            idx_j = pool_bool.indices[pool_bool.indptr[j]:pool_bool.indptr[j + 1]]
            if np.array_equal(idx_c, idx_j):
                n += 1
        res[k] = n
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--mol", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--topk", type=int, default=25)
    args = ap.parse_args()

    P = np.load(args.pred).astype(np.float32)
    mol_codes = np.load(args.mol).astype(np.int32)
    log(f"evaluar {args.tag}: P {P.shape}, {len(mol_codes):,} moléculas")
    pool, pool_nbits = load_pool(INTER)
    pool_bool = sparse.load_npz(INTER / "huellas_pool.npz").tocsr()
    Y_true = pool_bool[mol_codes].tocsr()
    n_sin_fp = int((np.diff(Y_true.indptr) == 0).sum())
    log(f"moléculas sin huella real: {n_sin_fp} (se excluyen del Tanimoto; "
        f"en retrieval su rango es 0 por construcción)")

    mask = np.diff(Y_true.indptr) > 0
    # a) Tanimoto
    t, tb = tanimoto_rows(P[mask], Y_true[mask])
    # b) nulo: huella media
    freq = np.load(INTER / "huellas_freq.npz")["freq"].astype(np.float32)
    Pn = np.tile(freq, (int(mask.sum()), 1))
    tn, tnb = tanimoto_rows(Pn, Y_true[mask])
    del Pn
    res = {
        "tag": args.tag, "n": int(len(mol_codes)), "n_con_huella": int(mask.sum()),
        "n_sin_huella": int(n_sin_fp),
        "tanimoto_cont_media": float(t.mean()), "tanimoto_cont_mediana": float(np.median(t)),
        "tanimoto_cont_p25": float(np.percentile(t, 25)), "tanimoto_cont_p75": float(np.percentile(t, 75)),
        "tanimoto_bin100_media": float(tb.mean()), "tanimoto_bin100_mediana": float(np.median(tb)),
        "nulo_tanimoto_cont_media": float(tn.mean()), "nulo_tanimoto_cont_mediana": float(np.median(tn)),
        "nulo_tanimoto_bin100_media": float(tnb.mean()),
    }
    log(f"Tanimoto continuo: media {t.mean():.4f} mediana {np.median(t):.4f} "
        f"(nulo: media {tn.mean():.4f} mediana {np.median(tn):.4f})")
    log(f"Tanimoto bin100: media {tb.mean():.4f} mediana {np.median(tb):.4f} "
        f"(nulo: {tnb.mean():.4f})")

    # c) recuperación
    top_idx, top_sim, ranks = retrieval(P, pool, pool_nbits, topk=args.topk,
                                        true_codes=mol_codes)
    m = mrr_from_ranks(ranks)
    res["retrieval"] = m
    log(f"retrieval mol-split: MRR@25 {m['mrr25']:.4f} Top-1 {m['top1']:.4f} "
        f"Top-5 {m['top5']:.4f} Recall@25 {m['recall25']:.4f}")
    np.save(VERIF / f"ranks_paso2_{args.tag}.npy", ranks)
    np.save(VERIF / f"top_idx_paso2_{args.tag}.npy", top_idx)

    # d) oráculo + colisiones (el oráculo solo depende del conjunto de
    # moléculas: se cachea por hash del array de códigos)
    import hashlib
    clave = hashlib.sha1(np.ascontiguousarray(mol_codes).tobytes()).hexdigest()[:12]
    cache_or = VERIF / f"ranks_oraculo_{clave}.npy"
    if cache_or.exists():
        ranks_or = np.load(cache_or)
        log(f"oráculo cargado de caché {cache_or.name}")
    else:
        Yd = Y_true.toarray().astype(np.float32)
        _, _, ranks_or = retrieval(Yd, pool, pool_nbits, topk=args.topk,
                                   true_codes=mol_codes)
        np.save(cache_or, ranks_or)
        del Yd
    m_or = mrr_from_ranks(ranks_or)
    res["oraculo_retrieval"] = m_or
    log(f"oráculo (huella real): MRR@25 {m_or['mrr25']:.4f} Top-1 {m_or['top1']:.4f} "
        f"Recall@25 {m_or['recall25']:.4f}")
    h_pool = hash_filas(pool_bool)
    col = colisiones(pool_bool, h_pool, mol_codes)
    res["colisiones_media"] = float(col.mean())
    res["colisiones_mediana"] = float(np.median(col))
    res["colisiones_frac_cero"] = float((col == 0).mean())
    res["colisiones_max"] = int(col.max())
    log(f"colisiones exactas de huella en el pool: media {col.mean():.2f}, "
        f"mediana {np.median(col):.0f}, fracción sin gemelas {(col==0).mean():.3f}, "
        f"máx {col.max()}")
    (VERIF / f"metricas_paso2_{args.tag}.json").write_text(json.dumps(res, indent=2))
    log(f"FIN evaluar {args.tag}")

if __name__ == "__main__":
    main()
