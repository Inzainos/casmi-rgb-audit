"""CASMI baseline 1 — 02_validar.py
Valida la búsqueda en librería con MRR@25.

- qmol (mol-split): las moléculas de validación NO están en la librería.
  El MRR@25 honesto aquí es 0 por construcción (la respuesta no existe en la
  librería); el valor de esta corrida es verificar el pipeline y medir
  tiempos/candidatos. Se reporta tal cual, sin maquillar.
- qspec (spec-split): de cada molécula se retuvieron ~3 espectros como
  consultas y el resto quedó en la librería. Es el estimador realista del
  rendimiento clase-1 (moléculas que SÍ están en la librería), que es lo que
  el envío final puede resolver.

Por molécula se agregan sus espectros tomando, por cada inchikey14 candidato,
la máxima similitud entre los espectros de consulta; el ranking es sobre
inchikey14 distintos (top 25). Métricas: MRR@25, top-1, top-5, recall@25,
y MRR@25 por ingest_lib mayoritario de la molécula.
"""
import argparse, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import sys
sys.path.insert(0, str(Path(__file__).parent))
from casmi_lib import (log, load_library, load_queries, row_stats, merge_bins,
                       score_query, rank_molecule, mrr_from_ranks, g, LN2, LN4)

BASE = Path("/home/hatch/workspace/snt-taxonomia/kaggle-casmi")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"


def self_check(X, prec, mol, rowsum, S_row, norm_cos):
    """Coseno y entropía de un espectro contra sí mismo (=1) y contra uno disjunto (~0)."""
    q = X[0]
    q_pmz = float(prec[0])
    # mini-librería: la fila 0 (idéntica) + una fila sin bins comunes
    q_bins = set(q.indices.tolist())
    disj = None
    for i in range(1, min(5000, X.shape[0])):
        if not (set(X[i].indices.tolist()) & q_bins):
            disj = i
            break
    assert disj is not None, "no se encontró espectro disjunto para el self-check"
    Xmini = sparse.vstack([X[0], X[disj]]).tocsr()
    prec_mini = np.array([q_pmz, q_pmz])
    mol_mini = np.array([mol[0], mol[disj]])
    rs_mini = rowsum[[0, disj]]
    S_mini = S_row[[0, disj]]
    nc_mini = norm_cos[[0, disj]]
    m, cos, ent = score_query(Xmini, prec_mini, mol_mini, rs_mini, S_mini, nc_mini,
                              q, q_pmz, "ppm")
    print(f"SELF-CHECK cos(idéntico)={cos[0]:.6f} ent(idéntico)={ent[0]:.6f} "
          f"ent(disjunto)={ent[1]:.6f} (esperado 1, 1, ~0)", flush=True)
    assert abs(cos[0] - 1.0) < 1e-5, cos[0]
    assert abs(ent[0] - 1.0) < 1e-5, ent[0]
    assert ent[1] < 0.05, ent[1]
    log("self-check OK")


def run_queries(X, prec, mol_sorted, rowsum, S_row, norm_cos,
                Q, q_prec, q_mol, q_lib, modo, tag):
    """Corre todas las consultas de un conjunto; devuelve DataFrame por molécula."""
    order = np.argsort(q_mol, kind="stable")
    q_mol_s = q_mol[order]
    uniq, starts = np.unique(q_mol_s, return_index=True)
    bounds = list(starts) + [len(q_mol_s)]
    rows = []
    t0 = time.time()
    n_cand_tot = 0
    n_q = 0
    for ui, um in enumerate(uniq):
        s, e = bounds[ui], bounds[ui + 1]
        idxs = order[s:e]
        best_cos, best_ent = [], []
        mols_cos, mols_ent = [], []
        for j in idxs:
            m, c, en = score_query(X, prec, mol_sorted, rowsum, S_row, norm_cos,
                                   Q[j], float(q_prec[j]), modo)
            n_q += 1
            if m is None:
                continue
            n_cand_tot += len(m)
            best_cos.append(c); mols_cos.append(m)
            best_ent.append(en); mols_ent.append(m)
        rank_cos = rank_ent = 0
        if best_cos:
            top_cos = rank_molecule(best_cos, mols_cos)[:25]
            if um in top_cos:
                rank_cos = int(top_cos.index(um)) + 1
            top_ent = rank_molecule(best_ent, mols_ent)[:25]
            if um in top_ent:
                rank_ent = int(top_ent.index(um)) + 1
        # ingest_lib mayoritario de la molécula
        libs, cnt = np.unique(q_lib[idxs], return_counts=True)
        lib_maj = str(libs[int(np.argmax(cnt))])
        rows.append({"mol": int(um), "ingest_lib": lib_maj, "n_espectros": int(e - s),
                     "rank_cos": rank_cos, "rank_ent": rank_ent})
        if (ui + 1) % 2000 == 0:
            log(f"{tag}/{modo}: {ui+1:,}/{len(uniq):,} moléculas "
                f"({(time.time()-t0):.0f}s, {n_q:,} espectros, {n_cand_tot:,} pares candidato)")
    df = pd.DataFrame(rows)
    log(f"{tag}/{modo}: {len(df):,} moléculas en {(time.time()-t0):.1f}s, "
        f"{n_q:,} espectros consulta, {n_cand_tot:,} pares candidato, "
        f"media candidatos/espectro {n_cand_tot/max(n_q,1):.1f}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", choices=["qmol", "qspec", "ambas"], default="ambas")
    ap.add_argument("--modos", choices=["ppm", "da", "ambos"], default="ambos")
    ap.add_argument("--binmerge", type=int, choices=[1, 5], default=1)
    ap.add_argument("--subset", type=int, default=0, help="nº moléculas por conjunto (0=todas)")
    ap.add_argument("--libtag", default="val_lib")
    args = ap.parse_args()

    X, prec, mol_sorted, lib_ing = load_library(INTER, args.libtag)
    if args.binmerge > 1:
        log(f"fusionando bins x{args.binmerge} (0.01 -> {0.01*args.binmerge:.2f} Da)")
        X = merge_bins(X, args.binmerge)
    rowsum, S_row, norm_cos = row_stats(X)
    log(f"row_stats: rowsum media {rowsum.mean():.2f}, S media {S_row.mean():.2f}")
    self_check(X, prec, mol_sorted, rowsum, S_row, norm_cos)

    lib_mols = set(np.unique(mol_sorted).tolist())
    modos = ["ppm", "da"] if args.modos == "ambos" else [args.modos]
    qtags = ["qmol", "qspec"] if args.queries == "ambas" else [args.queries]
    resumen = {}
    for qtag in qtags:
        Q, q_prec, q_mol, q_lib = load_queries(INTER, f"val_{qtag}")
        if args.binmerge > 1:
            Q = merge_bins(Q, args.binmerge)   # las consultas van en el mismo bineado que la librería
        q_mol = np.asarray(q_mol)
        # integridad del split
        q_mol_set = set(np.unique(q_mol).tolist())
        if qtag == "qmol":
            assert len(q_mol_set & lib_mols) == 0, "FALLO: moléculas de validación en la librería"
            log("integridad qmol OK: 0 moléculas de validación en la librería")
        else:
            assert q_mol_set <= lib_mols, "FALLO: moléculas spec-split fuera de la librería"
            log("integridad qspec OK: todas las moléculas de consulta están en la librería")
        if args.subset > 0:
            keep_mols = np.unique(q_mol)[:args.subset]
            keep = np.isin(q_mol, keep_mols)
            Q, q_prec, q_mol, q_lib = Q[keep], q_prec[keep], q_mol[keep], q_lib[keep]
            log(f"subset: {args.subset} moléculas de {qtag}")
        for modo in modos:
            df = run_queries(X, prec, mol_sorted, rowsum, S_row, norm_cos,
                             Q, q_prec, q_mol, q_lib, modo, qtag)
            suf = f"{qtag}_{modo}_bm{args.binmerge}"
            df.to_csv(VERIF / f"ranks_{suf}.csv", index=False)
            m_cos = mrr_from_ranks(df["rank_cos"].tolist())
            m_ent = mrr_from_ranks(df["rank_ent"].tolist())
            por_lib = {}
            for lib, g in df.groupby("ingest_lib"):
                if len(g) >= 30:
                    por_lib[str(lib)] = {"n": int(len(g)),
                                         "mrr25_cos": round(mrr_from_ranks(g["rank_cos"].tolist())["mrr25"], 4),
                                         "mrr25_ent": round(mrr_from_ranks(g["rank_ent"].tolist())["mrr25"], 4)}
            resumen[suf] = {"coseno": m_cos, "entropia": m_ent, "por_ingest_lib": por_lib}
            print(f"== {suf} ==\n  coseno: {m_cos}\n  entropía: {m_ent}", flush=True)
    (VERIF / f"validacion_bm{args.binmerge}.json").write_text(json.dumps(resumen, indent=2))
    log("VALIDACIÓN COMPLETA")


if __name__ == "__main__":
    main()
