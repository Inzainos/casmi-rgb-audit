"""CASMI ronda 2 — 22_rerank.py
WPC Idea 4: re-ranking Tanimoto del top-25 del lineal_bin.

1. Verifica que la huella Morgan recomputada con RDKit desde los SMILES del
   pool (mapa_smiles_full.parquet, mismo generador que 04_huellas.py: radio 2,
   2048 bits) coincide bit a bit con huellas_pool.npz para todos los
   candidatos del top-25.
2. Re-ordena el top-25 por Tanimoto entre la huella predicha CRUDA
   (clip [0,1], como en 15 sin --bin) y la huella del candidato, y mide
   MRR@25 después vs antes.
3. Control: re-ordenar por Tanimoto con la predicción BINARIZADA (top-100
   bits a 1, como en 15 --bin) debe reproducir el ranking original (es el
   ranking que ya hace 15 --bin); se reporta si el orden y el MRR coinciden.

Entrada: el top_idx que deja 15_evaluar_pred.py con --tag lineal_bin --bin:
  python3 15_evaluar_pred.py --pred intermedios/P_val_lineal.npy \
      --mol intermedios/P_val_mol_lineal.npy --tag lineal_bin --split mol --bin
Salida:
- verificacion/metricas_rerank_<tag>_val_mol.json
- verificacion/ranks_rerank_<tag>_val_mol.npy
Reglas: nada a Kaggle/GitHub; log en verificacion/; un solo proceso pesado.
"""
import argparse, time, json, sys, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:8.1f}s] {msg}", flush=True)

BASE = Path("/home/deamon/wt_deamonx/CASMI-sandbox/pruebas")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"
NBITS = 2048
RADIO = 2
TOPK = 25

_spec = importlib.util.spec_from_file_location(
    "ev13", BASE / "13_evaluar_masa.py")
_ev13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev13)
tanimoto_vec = _ev13.tanimoto_vec
sys.path.insert(0, str(BASE))
from casmi_lib import mrr_from_ranks  # media sobre TODAS las consultas
# (la v1 de este script promediaba 1/rango solo sobre los hits: inflaba el
# MRR por 1/recall; corregido 2026-11-10, ver metricas_rerank JSON)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lineal_bin")
    args = ap.parse_args()
    log(f"rerank {args.tag}: python {sys.version.split()[0]}, numpy {np.__version__}")

    # --- entradas ---
    top_idx = np.load(VERIF / f"top_idx_masa_{args.tag}_val_mol.npy")
    P = np.load(INTER / "P_val_lineal.npy", mmap_mode="r")
    codes_p = np.load(INTER / "P_val_mol_lineal.npy").astype(np.int32)
    qp = np.load(INTER / "query_prec_val_mol.npz")
    q_mol = qp["mol"].astype(np.int32)
    assert np.array_equal(codes_p, q_mol), "P_val_mol no coincide con query_prec"
    assert top_idx.shape[0] == len(q_mol), "top_idx no tiene 13,748 filas"
    n = len(q_mol)
    log(f"top_idx: {top_idx.shape}, P: {P.shape}, consultas: {n:,}")

    # nº real de candidatos por consulta (mismo filtro de masa que 15) para
    # recortar el top-25 real (el resto de la fila son ceros de relleno)
    idx = np.load(INTER / "precursores_full_idx.npz")
    prec_sorted, mol_sorted = idx["prec"], idx["mol"]
    q_prec, q_indptr = qp["prec"], qp["indptr"]
    cand_n = np.zeros(n, dtype=np.int32)
    for i in range(n):
        cands = set()
        for q in q_prec[q_indptr[i]:q_indptr[i + 1]]:
            tol = max(float(q) * 10.0 * 1e-6, 0.002)
            lo = int(np.searchsorted(prec_sorted, q - tol))
            hi = int(np.searchsorted(prec_sorted, q + tol))
            if hi > lo:
                cands.update(mol_sorted[lo:hi].tolist())
        cand_n[i] = len(cands)
    k_real = np.minimum(cand_n, TOPK)
    log(f"candidatos/mol: mediana {np.median(cand_n):.0f}, media {cand_n.mean():.1f}, "
        f"sin candidatos: {(cand_n == 0).sum()}, con <25: {(cand_n < 25).sum()}")

    # --- paso A: recomputar Morgan desde los SMILES y comparar con el pool ---
    pool = sparse.load_npz(INTER / "huellas_pool.npz").tocsr()
    pool_nbits = np.diff(pool.indptr).astype(np.float32)
    mol_index = np.load(VERIF / "mol_index.npy", allow_pickle=True)
    code_de_ik = {m: i for i, m in enumerate(mol_index)}
    mapa = pd.read_parquet(INTER / "mapa_smiles_full.parquet")
    sm_de_ik = dict(zip(mapa["inchikey14"], mapa["smiles_rep"]))
    union_codes = np.unique(np.concatenate(
        [top_idx[i, :k_real[i]] for i in range(n) if k_real[i] > 0]))
    log(f"unión de top-25: {len(union_codes):,} moléculas distintas")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIO, fpSize=NBITS)
    n_exact = 0
    n_sin_smiles = 0
    n_fallo = 0
    ejemplos_fallo = []
    for c in union_codes:
        ik = mol_index[c]
        sm = sm_de_ik.get(ik)
        if sm is None:
            n_sin_smiles += 1
            continue
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            n_fallo += 1
            if len(ejemplos_fallo) < 5:
                ejemplos_fallo.append((ik, sm))
            continue
        bits = set(int(b) for b in gen.GetFingerprint(mol).GetOnBits())
        pool_bits = set(pool.indices[pool.indptr[c]:pool.indptr[c + 1]].tolist())
        if bits == pool_bits:
            n_exact += 1
        elif len(ejemplos_fallo) < 5:
            ejemplos_fallo.append((ik, f"recomputado {len(bits)} bits, "
                                   f"pool {len(pool_bits)} bits, "
                                   f"dif simétrica {len(bits ^ pool_bits)}"))
    log(f"paso A: {n_exact:,}/{len(union_codes):,} huellas recomputadas idénticas "
        f"al pool; sin SMILES: {n_sin_smiles}, fallos de parseo: {n_fallo}")
    if ejemplos_fallo:
        log(f"ejemplos de diferencia/fallo: {ejemplos_fallo}")

    # --- MRR antes (desde el top_idx original) ---
    ranks_antes = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if k_real[i] == 0:
            continue
        hit = np.where(top_idx[i, :k_real[i]] == int(q_mol[i]))[0]
        ranks_antes[i] = int(hit[0] + 1) if len(hit) else 0
    m_antes = mrr_from_ranks(ranks_antes)
    log(f"antes (top-25 lineal_bin): MRR@25 {m_antes['mrr25']:.4f} "
        f"Top-1 {m_antes['top1']:.4f} Top-5 {m_antes['top5']:.4f} "
        f"Recall@25 {m_antes['recall25']:.4f}")

    # --- paso B: re-ordenar por Tanimoto con la predicción cruda ---
    ranks_crudo = np.zeros(n, dtype=np.int64)
    ranks_bin = np.zeros(n, dtype=np.int64)
    orden_igual = 0
    for i in range(n):
        if k_real[i] == 0:
            continue
        codes_c = top_idx[i, :k_real[i]].astype(np.int32)
        F = pool[codes_c].toarray().astype(np.float32)
        nb = pool_nbits[codes_c]
        p_raw = np.asarray(P[i], dtype=np.float32)
        p_crudo = np.clip(p_raw, 0.0, 1.0)
        p_bin = np.zeros(NBITS, dtype=np.float32)
        p_bin[np.argpartition(-np.clip(p_raw, 0.0, 1.0), 100)[:100]] = 1.0
        t_crudo = tanimoto_vec(p_crudo, F, nb)
        t_bin = tanimoto_vec(p_bin, F, nb)
        ord_crudo = codes_c[np.argsort(-t_crudo, kind="stable")]
        ord_bin = codes_c[np.argsort(-t_bin, kind="stable")]
        if np.array_equal(ord_bin, codes_c):
            orden_igual += 1
        for ranks, orden in ((ranks_crudo, ord_crudo), (ranks_bin, ord_bin)):
            hit = np.where(orden == int(q_mol[i]))[0]
            ranks[i] = int(hit[0] + 1) if len(hit) else 0
        if (i + 1) % 4000 == 0:
            log(f"  rerank: {i+1:,}/{n:,}")
    m_crudo = mrr_from_ranks(ranks_crudo)
    m_bin = mrr_from_ranks(ranks_bin)
    log(f"paso B (re-rank por Tanimoto crudo): MRR@25 {m_crudo['mrr25']:.4f} "
        f"Top-1 {m_crudo['top1']:.4f} Top-5 {m_crudo['top5']:.4f} "
        f"Recall@25 {m_crudo['recall25']:.4f}")
    log(f"control (re-rank por Tanimoto bin): MRR@25 {m_bin['mrr25']:.4f}; "
        f"orden idéntico al original en {orden_igual:,}/{int((k_real > 0).sum()):,} consultas")

    out = {"tag": args.tag, "split": "val_mol", "n": int(n),
           "mrr_definicion": "media de 1/rango sobre TODAS las consultas "
                             "(0 si fuera del top-25), como casmi_lib",
           "mrr_corregido_2026_11_10": True,
           "pasoA_recomputadas": int(len(union_codes)),
           "pasoA_exactas_pool": int(n_exact),
           "pasoA_sin_smiles": int(n_sin_smiles),
           "pasoA_fallo_parseo": int(n_fallo),
           "pasoA_ejemplos_diferencia": [str(e) for e in ejemplos_fallo],
           "antes_mrr25": m_antes["mrr25"], "antes_top1": m_antes["top1"],
           "antes_top5": m_antes["top5"], "antes_recall25": m_antes["recall25"],
           "rerank_crudo_mrr25": m_crudo["mrr25"],
           "rerank_crudo_top1": m_crudo["top1"],
           "rerank_crudo_top5": m_crudo["top5"],
           "rerank_crudo_recall25": m_crudo["recall25"],
           "control_bin_mrr25": m_bin["mrr25"],
           "control_bin_orden_identico": int(orden_igual),
           "control_bin_n_con_candidatos": int((k_real > 0).sum())}
    np.save(VERIF / f"ranks_rerank_{args.tag}_val_mol.npy", ranks_crudo)
    (VERIF / f"metricas_rerank_{args.tag}_val_mol.json").write_text(
        json.dumps(out, indent=2))
    log(f"FIN rerank {args.tag}: antes {m_antes['mrr25']:.4f} -> "
        f"crudo {m_crudo['mrr25']:.4f} (control bin {m_bin['mrr25']:.4f})")

if __name__ == "__main__":
    main()
