"""CASMI paso 3 — 15_evaluar_pred.py
Evalúa UNA predicción de huellas (P .npy, moléculas x 2048) con el filtro
de masa de precursor (mismo protocolo que 13_evaluar_masa.py: candidatos
= moléculas con algún espectro en full_lib dentro de ±10 ppm, piso
0.002 Da, de cualquiera de los precursores de la consulta; ranking por
Tanimoto contra la huella predicha, top-25, MRR@25).

Reutiliza tanimoto_vec/topk_de de 13 y mrr_from_ranks/log de casmi_lib.
--bin aplica la variante lineal_bin (clip [0,1] + top-100 bits a 1).

Uso:
  python3 15_evaluar_pred.py --pred intermedios/P_val_ctrl20k.npy \
      --mol intermedios/P_val_mol_ctrl20k.npy --tag ctrl20k --split mol
  python3 15_evaluar_pred.py --pred intermedios/P_val_bin005.npy \
      --mol intermedios/P_val_mol_bin005.npy --tag bin005 --split mol --bin
Salida: verificacion/metricas_masa_<tag>_<split>.json (+ ranks/top_idx).
Reglas: nada a Kaggle/GitHub; log en verificacion/; un solo proceso pesado.
"""
import argparse, time, json, sys, importlib.util
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).parent))
from casmi_lib import log, mrr_from_ranks
_spec = importlib.util.spec_from_file_location(
    "ev13", Path(__file__).parent / "13_evaluar_masa.py")
_ev13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev13)
tanimoto_vec, topk_de = _ev13.tanimoto_vec, _ev13.topk_de

BASE = Path("/home/hatch/workspace/enveda-casmi-2026")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"
TOPK = 25

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--mol", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", choices=["mol", "spec"], default="mol")
    ap.add_argument("--bin", action="store_true",
                    help="clip [0,1] + top-100 bits a 1 (variante lineal_bin)")
    ap.add_argument("--tol-ppm", type=float, default=10.0)
    ap.add_argument("--tol-min", type=float, default=0.002)
    args = ap.parse_args()
    tag = "val_mol" if args.split == "mol" else "val_spec"

    idx = np.load(INTER / "precursores_full_idx.npz")
    prec_sorted, mol_sorted = idx["prec"], idx["mol"]
    qp = np.load(INTER / f"query_prec_{tag}.npz")
    q_mol, q_prec, q_indptr = qp["mol"].astype(np.int32), qp["prec"], qp["indptr"]
    pool = sparse.load_npz(INTER / "huellas_pool.npz").tocsr()
    pool_nbits = np.diff(pool.indptr).astype(np.float32)
    Y_true = pool[q_mol].tocsr()
    P = np.load(args.pred, mmap_mode="r")
    codes = np.load(args.mol).astype(np.int32)
    assert P.shape[0] == len(q_mol), (P.shape, len(q_mol))
    assert np.array_equal(codes, q_mol), "los códigos P_mol no coinciden con query_prec"
    log(f"evaluar pred {args.tag} ({'bin' if args.bin else 'crudo'}) {tag}: "
        f"{len(q_mol):,} moléculas, P {P.shape} (memmap)")

    n = len(q_mol)
    ranks = np.zeros(n, dtype=np.int64)
    top_idx = np.zeros((n, TOPK), dtype=np.int32)
    t_cont = np.zeros(n, dtype=np.float32)
    cand_n = np.zeros(n, dtype=np.int32)
    t0 = time.time()
    for i in range(n):
        cands = set()
        for q in q_prec[q_indptr[i]:q_indptr[i + 1]]:
            tol = max(float(q) * args.tol_ppm * 1e-6, args.tol_min)
            lo = int(np.searchsorted(prec_sorted, q - tol))
            hi = int(np.searchsorted(prec_sorted, q + tol))
            if hi > lo:
                cands.update(mol_sorted[lo:hi].tolist())
        codes_c = np.fromiter(cands, dtype=np.int32)
        cand_n[i] = len(codes_c)
        if len(codes_c) == 0:
            continue
        F = pool[codes_c].toarray().astype(np.float32)
        nb = pool_nbits[codes_c]
        y_row = Y_true[i].toarray().ravel().astype(np.float32)
        true_code = int(q_mol[i])
        p_raw = np.asarray(P[i], dtype=np.float32)
        if args.bin:
            p = np.zeros(2048, dtype=np.float32)
            p[np.argpartition(-np.clip(p_raw, 0.0, 1.0), 100)[:100]] = 1.0
        else:
            p = np.clip(p_raw, 0.0, 1.0)
        t = tanimoto_vec(p, F, nb)
        top_c, _ = topk_de(t, codes_c)
        top_idx[i, :len(top_c)] = top_c
        hit = np.where(top_c == true_code)[0]
        ranks[i] = int(hit[0] + 1) if len(hit) else 0
        d = float(p @ y_row)
        den = float(p @ p) + float(y_row @ y_row) - d
        t_cont[i] = d / den if den > 0 else 0.0
        if (i + 1) % 4000 == 0:
            log(f"  {args.tag} {tag}: {i+1:,}/{n:,} ({time.time()-t0:.0f}s, "
                f"cand mediana {np.median(cand_n[:i+1]):.0f})")
    log(f"retrieval completo: {n:,} consultas en {time.time()-t0:.0f}s, "
        f"candidatos/mol mediana {np.median(cand_n):.0f} media {cand_n.mean():.1f}")

    m = mrr_from_ranks(ranks)
    out = {"tag": args.tag, "split": tag, "bin": bool(args.bin),
           "tol_ppm": args.tol_ppm, "tol_min": args.tol_min, "n": int(n),
           "cand_mediana": float(np.median(cand_n)), "cand_media": float(cand_n.mean()),
           "mrr25": m["mrr25"], "top1": m["top1"], "top5": m["top5"],
           "recall25": m["recall25"],
           "rank_mediana_presentes": m["rank_mediana_presentes"],
           "tanimoto_cont_media": float(t_cont.mean()),
           "tanimoto_cont_mediana": float(np.median(t_cont))}
    np.save(VERIF / f"ranks_masa_{args.tag}_{tag}.npy", ranks)
    np.save(VERIF / f"top_idx_masa_{args.tag}_{tag}.npy", top_idx)
    (VERIF / f"metricas_masa_{args.tag}_{tag}.json").write_text(json.dumps(out, indent=2))
    log(f"  {args.tag} ({'bin' if args.bin else 'crudo'}): MRR@25 {m['mrr25']:.4f} "
        f"Top-1 {m['top1']:.4f} Top-5 {m['top5']:.4f} Recall@25 {m['recall25']:.4f} "
        f"Tanimoto {t_cont.mean():.4f}")
    log(f"FIN evaluar pred {args.tag}")

if __name__ == "__main__":
    main()
