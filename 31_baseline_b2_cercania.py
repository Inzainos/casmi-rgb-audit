#!/usr/bin/env python3
"""CASMI — Baseline B2 (protocolo generativo de novo): piso de cercanía.

Mide, sobre las 13,748 moléculas val evaluables, lo cerca que queda el
retrieval de Morgan (lineal_bin, MRR@25 mol-split 0.4146 INFLADO por
verdad-en-pool) de la verdad, en similitud de Tanimoto (Morgan 2048,
radio 2, las mismas huellas de paso 2):

- Tanimoto@1: similitud entre el top-1 predicho y la verdad.
- Mejor-en-25: máxima similitud entre los 25 predichos y la verdad.
- Tasa de "cercanos": fracción con Tanimoto@1 >= 0.675 (mismo umbral
  que MS2Mol reporta en EnvedaDark, ver protocolo).

Esto es el PISO que un generativo debe superar en la métrica secundaria
(cercanía estructural): el retrieval, aun estando equivocado en exacto,
devuelve moléculas parecidas. El MRR@25 exacto del retrieval en el
protocolo honesto (pool SIN consultas) es 0 por construcción y ya está
medido: baseline-1 qmol MRR 0.0000, n=13,748 (verificacion/
validacion_bm1.json) — ese es el control B0.

Verificación interna (sin suponer el orden de filas):
- Las filas de top_idx_masa_val_mol_lineal_bin.npy deben estar en el
  orden de val_mols ordenado. Se comprueba con las filas de rank 1:
  en ellas top_idx[fila, 0] DEBE ser el código de la verdad. Si alguna
  falla, se aborta: el orden no es el supuesto.
- Además se comprueba que ningún top-1 es la verdad cuando rank != 1.

Solo lee. Escribe verificacion/baseline_b2_cercania.json y su log.
No entrena nada. No toca Kaggle ni GitHub.
"""
import json
import time
import resource
from pathlib import Path

import numpy as np
from scipy import sparse

T0 = time.time()


def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time() - T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)


BASE = Path("/home/hatch/workspace/enveda-casmi-2026")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"
UMBRAL_CERCANO = 0.675


def tanimoto_csr(fp_csr, idx_a, idx_b):
    """Tanimoto entre dos filas de una CSR booleana, vectorizado por bloques."""
    a = fp_csr[idx_a]
    b = fp_csr[idx_b]
    inter = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    na = np.asarray(a.sum(axis=1)).ravel()
    nb = np.asarray(b.sum(axis=1)).ravel()
    union = na + nb - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def main():
    log("B2: piso de cercanía del retrieval (lineal_bin) en val_mol")
    z = np.load(VERIF / "split_arrays.npz", allow_pickle=True)
    mols = z["mols"]
    val_mols = z["val_mols"]
    log(f"moléculas totales {len(mols):,} | val {len(val_mols):,}")
    mol_index = {m: i for i, m in enumerate(mols.tolist())}
    val_codes = np.array([mol_index[m] for m in val_mols.tolist()], dtype=np.int64)

    top_idx = np.load(VERIF / "top_idx_masa_val_mol_lineal_bin.npy")
    ranks = np.load(VERIF / "ranks_masa_val_mol_lineal_bin.npy")
    log(f"top_idx {top_idx.shape} {top_idx.dtype} | ranks {ranks.shape} "
        f"| ranks>0: {(ranks > 0).sum():,} | rank==1: {(ranks == 1).sum():,}")
    assert top_idx.shape[0] == len(val_mols) or top_idx.shape[0] == 13748, (
        f"n filas top_idx {top_idx.shape[0]} no coincide con val "
        f"{len(val_mols)} ni con 13,748 evaluables")
    n = top_idx.shape[0]
    # Mapeo exacto fila->verdad, sin suponer el orden: las consultas val
    # están en intermedios/val_qmol_meta.npz (campo "mol": código de
    # molécula por espectro de consulta). Las filas de 13_evaluar_masa.py
    # se producen en orden de molécula creciente (uniq sorted de
    # 02_validar.run_queries), así que la fila f corresponde a la
    # f-ésima molécula distinta de val_qmol. Las 42 moléculas val sin
    # fila son las que no tienen ningún espectro de consulta válido
    # (13,790 - 13,748).
    qmeta = np.load(INTER / "val_qmol_meta.npz", allow_pickle=True)
    q_mol = qmeta["mol"]
    truth_full = np.unique(q_mol.astype(np.int64))
    log(f"consultas val_qmol: {len(q_mol):,} espectros, "
        f"{len(truth_full):,} moléculas distintas")
    assert len(truth_full) == n, (
        f"moléculas distintas en val_qmol {len(truth_full)} != filas "
        f"top_idx {n}")
    assert np.all(np.isin(truth_full, val_codes)), \
        "una molécula de consulta val no está en val_mols"
    # Verificación cruzada: en filas rank>0 la verdad debe ser
    # top_idx[f, rank-1] y coincidir con truth_full[f].
    rpos = np.where(ranks > 0)[0]
    truth_from_rank = np.array([int(top_idx[f, int(ranks[f]) - 1]) for f in rpos])
    assert np.array_equal(truth_from_rank, truth_full[rpos]), (
        "la verdad por rank no coincide con val_qmol: orden incorrecto")
    log(f"mapeo fila->verdad VERIFICADO por dos vías independientes "
        f"({len(rpos):,} filas rank>0 coinciden con val_qmol)")
    known = truth_full[ranks > 0]
    log(f"verdades conocidas por rank>0: {len(known):,} únicas")
    n_r0 = int((ranks == 0).sum())
    log(f"filas rank 0 (verdad fuera del top-25): {n_r0:,}")
    log(f"moléculas val sin fila (sin espectros válidos): "
        f"{len(val_mols) - n} (13,790 - 13,748)")

    # Huellas del pool (todas las moléculas de train, CSR booleana).
    h = np.load(INTER / "huellas_pool.npz")
    fp = sparse.csr_matrix((h["data"], h["indices"], h["indptr"]),
                           shape=tuple(h["shape"]))
    log(f"huellas_pool {fp.shape} nnz {fp.nnz:,}")
    # Sanidad: la verdad debe tener huella (bits>0); si alguna verdad no
    # tiene huella (molécula sin SMILES válido), se marca y se excluye del
    # cómputo de Tanimoto con su motivo.
    nbits_truth = np.asarray(fp[truth_full].sum(axis=1)).ravel()
    sin_huella = nbits_truth == 0
    log(f"verdades sin huella en el pool: {int(sin_huella.sum())} / {n}")
    ok = ~sin_huella
    top1 = top_idx[:, 0]
    # Tanimoto@1 (todas las filas con huella en la verdad).
    t1_sim = tanimoto_csr(fp, top1[ok], truth_full[ok])
    # Mejor-en-25: máximo sobre los 25 candidatos, por bloques.
    best = np.zeros(int(ok.sum()), dtype=np.float64)
    idx_ok = np.where(ok)[0]
    for k in range(25):
        sim_k = tanimoto_csr(fp, top_idx[idx_ok, k], truth_full[idx_ok])
        best = np.maximum(best, sim_k)
    # --- Variante HONESTA (pool SIN consultas): la verdad no puede ser
    # candidata. El top-1 honesto es el mejor candidato distinto de la
    # verdad; el mejor-en-25 honesto excluye la verdad. En filas rank==1
    # el top-1 honesto es top_idx[f,1]; en el resto es top_idx[f,0].
    # (La verdad solo puede estar una vez en el top-25: un SMILES por
    # inchikey14 en el ranking.)
    top1_honest = np.where(ranks == 1, top_idx[:, 1], top_idx[:, 0])
    assert np.all(top1_honest != truth_full), "el top-1 honesto sigue siendo la verdad en alguna fila"
    t1h_sim = tanimoto_csr(fp, top1_honest[ok], truth_full[ok])
    best_h = np.zeros(int(ok.sum()), dtype=np.float64)
    for k in range(25):
        cand = top_idx[idx_ok, k]
        sim_k = tanimoto_csr(fp, cand, truth_full[idx_ok])
        sim_k = np.where(cand == truth_full[idx_ok], -1.0, sim_k)
        best_h = np.maximum(best_h, sim_k)
    assert np.all(best_h >= 0), "mejor-en-25 honesto negativo: todas las similitudes excluidas"
    # Fracción de top-1 que ES la verdad (debe coincidir con top1 del
    # modelo: filas rank==1).
    top1_es_verdad = (top1 == truth_full)
    log(f"top-1 == verdad: {int(top1_es_verdad.sum()):,} "
        f"(rank==1: {int((ranks == 1).sum()):,}) — deben coincidir")
    assert int(top1_es_verdad.sum()) == int((ranks == 1).sum()), \
        "top-1==verdad no coincide con rank==1: orden o mapeo incorrecto"

    res = {
        "modelo": "lineal_bin (Morgan, filtro masa +-10ppm)",
        "n_val_evaluables": int(n),
        "n_val_split": int(len(val_mols)),
        "n_sin_huella_verdad": int(sin_huella.sum()),
        "n_con_huella": int(ok.sum()),
        "tanimoto_top1_media": float(t1_sim.mean()),
        "tanimoto_top1_mediana": float(np.median(t1_sim)),
        "tanimoto_top1_p90": float(np.percentile(t1_sim, 90)),
        "tanimoto_top1_frac_ge_0.675": float((t1_sim >= UMBRAL_CERCANO).mean()),
        "tanimoto_top1_frac_ge_0.5": float((t1_sim >= 0.5).mean()),
        "tanimoto_mejor25_media": float(best.mean()),
        "tanimoto_mejor25_mediana": float(np.median(best)),
        "tanimoto_mejor25_p90": float(np.percentile(best, 90)),
        "tanimoto_mejor25_frac_ge_0.675": float((best >= UMBRAL_CERCANO).mean()),
        "top1_es_verdad": int(top1_es_verdad.sum()),
        "umbral_cercano": UMBRAL_CERCANO,
        "honesto_sin_verdad_en_pool": {
            "tanimoto_top1_media": float(t1h_sim.mean()),
            "tanimoto_top1_mediana": float(np.median(t1h_sim)),
            "tanimoto_top1_p90": float(np.percentile(t1h_sim, 90)),
            "tanimoto_top1_frac_ge_0.675": float((t1h_sim >= UMBRAL_CERCANO).mean()),
            "tanimoto_top1_frac_ge_0.5": float((t1h_sim >= 0.5).mean()),
            "tanimoto_mejor25_media": float(best_h.mean()),
            "tanimoto_mejor25_mediana": float(np.median(best_h)),
            "tanimoto_mejor25_p90": float(np.percentile(best_h, 90)),
            "tanimoto_mejor25_frac_ge_0.675": float((best_h >= UMBRAL_CERCANO).mean()),
        },
    }
    out = VERIF / "baseline_b2_cercania.json"
    out.write_text(json.dumps(res, indent=2))
    log(f"RESULTADO B2: {json.dumps(res, indent=2)}")
    log(f"escrito {out}")
    log("FIN B2 — sin entrenar nada, solo lectura")


if __name__ == "__main__":
    main()
