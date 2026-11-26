"""CASMI paso 3 — 13_evaluar_masa.py
Retrieval de huellas con filtro por masa de precursor (ver 12_masas.py).

Por molécula de consulta: candidatos = unión de moléculas con algún
espectro en full_lib dentro de ±10 ppm (piso 0.002 Da) de cualquiera de
sus precursores. Se ordenan solo esos candidatos por Tanimoto contra la
huella predicha. Variantes evaluadas en una sola pasada (las P se leen
por memmap, una fila a la vez; el pool se lee por filas candidatas):
  nulo      huella media (frecuencias) — control: ¿el filtro solo ya gana?
  oraculo   huella real — techo con este filtro y este pool
  knn       P_val_knn (mejor paso 2: MRR 0.0346 sin filtro)
  lineal    P_val_lineal (MRR 0.0085 sin filtro)
  lineal_bin lineal clip [0,1] + top-100 bits a 1 — el bin100 era su fuerte
  topo_v2   P_val_topo_v2 (MRR 0.0148 sin filtro)
  ens       media de knn+lineal+topo_v2 (clip [0,1])
  rrf       fusión por rango recíproco de los top-25 de knn/lineal/topo_v2
Uso: python3 13_evaluar_masa.py --split mol|spec [--tol-ppm 10] [--tol-min 0.002]
"""
import argparse, time, resource, json
from pathlib import Path
import numpy as np
from scipy import sparse
import sys
sys.path.insert(0, str(Path(__file__).parent))
from casmi_lib import log, mrr_from_ranks

BASE = Path("/home/hatch/workspace/enveda-casmi-2026")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"
TOPK = 25


def tanimoto_vec(p, F, nbits_c):
    """Tanimoto continuo entre p (2048,) y las filas de F (C,2048) float32.
    Devuelve array (C,) float32."""
    dots = F @ p
    p2 = float(p @ p)
    den = p2 + nbits_c - dots
    t = np.zeros(len(dots), dtype=np.float32)
    m = den > 0
    t[m] = dots[m] / den[m]
    return t


def topk_de(t, codes, k=TOPK):
    """Top-k índices de t (desc) y sus códigos."""
    if len(t) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    kk = min(k, len(t))
    if kk < len(t):
        part = np.argpartition(-t, kk - 1)[:kk]
    else:
        part = np.arange(len(t))
    part = part[np.argsort(-t[part])]
    return codes[part].astype(np.int32), t[part].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["mol", "spec"], default="mol")
    ap.add_argument("--tol-ppm", type=float, default=10.0)
    ap.add_argument("--tol-min", type=float, default=0.002)
    args = ap.parse_args()
    tag = "val_mol" if args.split == "mol" else "val_spec"
    ptag = "val" if args.split == "mol" else "spec"

    idx = np.load(INTER / "precursores_full_idx.npz")
    prec_sorted, mol_sorted = idx["prec"], idx["mol"]
    qp = np.load(INTER / f"query_prec_{tag}.npz")
    q_mol, q_prec, q_indptr = qp["mol"].astype(np.int32), qp["prec"], qp["indptr"]
    pool = sparse.load_npz(INTER / "huellas_pool.npz").tocsr()
    pool_nbits = np.diff(pool.indptr).astype(np.float32)
    freq = np.load(INTER / "huellas_freq.npz")["freq"].astype(np.float32)
    Y_true = pool[q_mol].tocsr()
    log(f"evaluar masa {tag}: {len(q_mol):,} moléculas, pool {pool.shape[0]:,}")

    P = {
        "knn": np.load(INTER / f"P_{ptag}_knn.npy", mmap_mode="r"),
        "lineal": np.load(INTER / f"P_{ptag}_lineal.npy", mmap_mode="r"),
        "topo_v2": np.load(INTER / f"P_{ptag}_topo_v2.npy", mmap_mode="r"),
    }
    for k, v in P.items():
        assert v.shape[0] == len(q_mol), (k, v.shape, len(q_mol))
        log(f"  P {k}: {v.shape} (memmap)")
    codes_knn = np.load(INTER / f"P_{ptag}_mol_knn.npy").astype(np.int32)
    assert np.array_equal(codes_knn, q_mol), "los códigos P_mol no coinciden con query_prec"
    log("  códigos P_mol verificados contra query_prec")

    variants = ["nulo", "oraculo", "knn", "lineal", "lineal_bin", "topo_v2", "ens", "rrf"]
    n = len(q_mol)
    ranks = {v: np.zeros(n, dtype=np.int64) for v in variants}
    top_idx = {v: np.zeros((n, TOPK), dtype=np.int32) for v in variants}
    t_cont = {v: np.zeros(n, dtype=np.float32) for v in variants if v != "rrf"}
    cand_n = np.zeros(n, dtype=np.int32)
    t0 = time.time()
    for i in range(n):
        # candidatos por masa (unión de ventanas de sus precursores)
        cands = set()
        for q in q_prec[q_indptr[i]:q_indptr[i + 1]]:
            tol = max(float(q) * args.tol_ppm * 1e-6, args.tol_min)
            lo = int(np.searchsorted(prec_sorted, q - tol))
            hi = int(np.searchsorted(prec_sorted, q + tol))
            if hi > lo:
                cands.update(mol_sorted[lo:hi].tolist())
        codes = np.fromiter(cands, dtype=np.int32)
        cand_n[i] = len(codes)
        if len(codes) == 0:
            continue
        F = pool[codes].toarray().astype(np.float32)
        nb = pool_nbits[codes]
        y_row = Y_true[i].toarray().ravel().astype(np.float32)
        true_code = int(q_mol[i])
        # vectores predichos por variante
        p_knn = np.clip(np.asarray(P["knn"][i], dtype=np.float32), 0.0, 1.0)
        p_lin_raw = np.asarray(P["lineal"][i], dtype=np.float32)
        p_lin = np.clip(p_lin_raw, 0.0, 1.0)
        p_topo = np.clip(np.asarray(P["topo_v2"][i], dtype=np.float32), 0.0, 1.0)
        p_lin_bin = np.zeros(2048, dtype=np.float32)
        p_lin_bin[np.argpartition(-p_lin_raw, 100)[:100]] = 1.0
        p_ens = (p_knn + p_lin + p_topo) / 3.0
        preds = {"knn": p_knn, "lineal": p_lin, "lineal_bin": p_lin_bin,
                 "topo_v2": p_topo, "ens": p_ens, "nulo": freq,
                 "oraculo": y_row}
        tops = {}
        for v, p in preds.items():
            t = tanimoto_vec(p, F, nb)
            tops[v] = topk_de(t, codes)
            top_idx[v][i, :len(tops[v][0])] = tops[v][0]
            hit = np.where(tops[v][0] == true_code)[0]
            ranks[v][i] = int(hit[0] + 1) if len(hit) else 0
            # Tanimoto contra la huella real (misma fórmula, 1 fila)
            d = float(p @ y_row)
            den = float(p @ p) + float(y_row @ y_row) - d
            t_cont[v][i] = d / den if den > 0 else 0.0
        # RRF sobre los top-25 de knn/lineal/topo_v2
        score = {}
        for v in ("knn", "lineal", "topo_v2"):
            for r, c in enumerate(tops[v][0]):
                score[int(c)] = score.get(int(c), 0.0) + 1.0 / (60.0 + r + 1)
        order = sorted(score.items(), key=lambda kv: -kv[1])[:TOPK]
        top_rrf = np.asarray([c for c, _ in order], dtype=np.int32)
        top_idx["rrf"][i, :len(top_rrf)] = top_rrf
        hit = np.where(top_rrf == true_code)[0]
        ranks["rrf"][i] = int(hit[0] + 1) if len(hit) else 0
        if (i + 1) % 2000 == 0:
            log(f"  masa {tag}: {i+1:,}/{n:,} ({time.time()-t0:.0f}s, "
                f"cand mediana {np.median(cand_n[:i+1]):.0f})")
    log(f"retrieval con masa completo: {n:,} consultas en {time.time()-t0:.0f}s, "
        f"candidatos/mol mediana {np.median(cand_n):.0f} media {cand_n.mean():.1f}")

    out = {"split": tag, "tol_ppm": args.tol_ppm, "tol_min": args.tol_min,
           "n": int(n), "cand_mediana": float(np.median(cand_n)),
           "cand_media": float(cand_n.mean())}
    for v in variants:
        m = mrr_from_ranks(ranks[v])
        np.save(VERIF / f"ranks_masa_{tag}_{v}.npy", ranks[v])
        np.save(VERIF / f"top_idx_masa_{tag}_{v}.npy", top_idx[v])
        d = {"mrr25": m["mrr25"], "top1": m["top1"], "top5": m["top5"],
             "recall25": m["recall25"],
             "rank_mediana_presentes": m["rank_mediana_presentes"]}
        if v in t_cont:
            d["tanimoto_cont_media"] = float(t_cont[v].mean())
            d["tanimoto_cont_mediana"] = float(np.median(t_cont[v]))
        out[v] = d
        log(f"  {v}: MRR@25 {m['mrr25']:.4f} Top-1 {m['top1']:.4f} "
            f"Top-5 {m['top5']:.4f} Recall@25 {m['recall25']:.4f}"
            + (f" Tanimoto {t_cont[v].mean():.4f}" if v in t_cont else ""))
    (VERIF / f"metricas_masa_{tag}.json").write_text(json.dumps(out, indent=2))
    log(f"FIN evaluar masa {tag}")

if __name__ == "__main__":
    main()
