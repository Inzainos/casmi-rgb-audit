"""CASMI baseline 1 — 03_predecir_test.py
Genera salida/submission.csv para el test real con la librería completa.

- Bineado idéntico al de la construcción (top-100 picos, bins 0.01 Da).
- Por espectro de test: candidatos por precursor (ventana --modo) y coseno.
- Por molecule_id: se agregan sus espectros (máxima similitud por inchikey14
  candidato) y se devuelven hasta 25 SMILES distintos (uno por inchikey14,
  el SMILES representativo de train), en orden de similitud, separados por ';'.
- NO envía nada a Kaggle. Verifica formato contra sample_submission.csv.
"""
import argparse, time, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
sys.path.insert(0, str(Path(__file__).parent))
from casmi_lib import (log, load_library, row_stats, merge_bins, score_query,
                       rank_molecule)
from importlib import import_module
_construir = import_module("01_construir")

BASE = Path("/home/hatch/workspace/snt-taxonomia/kaggle-casmi")
DATOS = BASE / "datos"
INTER = BASE / "intermedios"
SALIDA = BASE / "salida"
VERIF = BASE / "verificacion"
SALIDA.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["ppm", "da"], default="ppm")
    ap.add_argument("--binmerge", type=int, choices=[1, 5], default=1)
    ap.add_argument("--libtag", default="full_lib")
    args = ap.parse_args()

    X, prec, mol_sorted, _ = load_library(INTER, args.libtag)
    if args.binmerge > 1:
        log(f"fusionando bins x{args.binmerge}")
        X = merge_bins(X, args.binmerge)
    rowsum, S_row, norm_cos = row_stats(X)

    mapa = pd.read_parquet(INTER / "mapa_smiles_full.parquet")
    mol_names = np.load(VERIF / "mol_index.npy", allow_pickle=True)
    mol_code = {m: i for i, m in enumerate(mol_names)}
    rep = dict(zip(mapa["inchikey14"], mapa["smiles_rep"]))
    # El mapa cubre las moléculas con >=1 espectro que pasó los filtros; las
    # moléculas sin ningún espectro válido no pueden ser candidatas (no están
    # en la librería). Lo que sí debe cumplirse: toda molécula DE LA LIBRERÍA
    # tiene SMILES representativo.
    lib_codes = set(np.unique(mol_sorted).tolist())
    missing = [c for c in lib_codes if mol_names[c] not in rep]
    assert len(missing) == 0, f"{len(missing)} moléculas de la librería sin SMILES"
    log(f"mapa SMILES: {len(rep):,} moléculas de {len(mol_names):,} "
        f"({len(mol_names)-len(rep):,} sin ningún espectro válido; no pueden ser candidatas)")

    # ---- binear test ----
    test = pd.read_parquet(DATOS / "test.parquet")
    rows, cols, data = [], [], []
    q_mol_id, q_pmz = [], []
    n_drop = 0
    for i, r in test.iterrows():
        pmz = float(r["precursor_mz"])
        b = _construir.binear_fila(r["ms2_mzs"], r["ms2_normalized_intensities"])
        if b is None or not (100.0 <= pmz <= 2000.0):
            n_drop += 1
            continue
        bins, vals = b
        k = len(q_mol_id)
        rows.extend([k] * len(bins))
        cols.extend(bins.tolist())
        data.extend(vals.tolist())
        q_mol_id.append(r["molecule_id"])
        q_pmz.append(pmz)
    Q = sparse.csr_matrix((np.asarray(data, dtype=np.float32),
                           (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
                          shape=(len(q_mol_id), 200_000))
    Q.sum_duplicates()
    if args.binmerge > 1:
        Q = merge_bins(Q, args.binmerge)   # mismo bineado que la librería
    q_mol_id = np.asarray(q_mol_id, dtype=object)
    q_pmz = np.asarray(q_pmz, dtype=np.float64)
    log(f"test bineado: {Q.shape[0]:,} espectros, {Q.nnz:,} nnz, descartados {n_drop}")

    # ---- búsqueda por molecule_id ----
    order = np.argsort(q_mol_id, kind="stable")
    q_mol_s = q_mol_id[order]
    uniq, starts = np.unique(q_mol_s, return_index=True)
    bounds = list(starts) + [len(q_mol_s)]
    out_rows = []
    det = []
    t0 = time.time()
    for ui, mid in enumerate(uniq):
        s, e = bounds[ui], bounds[ui + 1]
        best, mols = [], []
        n_cand = 0
        for j in order[s:e]:
            m, c, _en = score_query(X, prec, mol_sorted, rowsum, S_row, norm_cos,
                                    Q[j], float(q_pmz[j]), args.modo)
            if m is None:
                continue
            n_cand += len(m)
            best.append(c); mols.append(m)
        if not best:
            out_rows.append({"molecule_id": mid, "smiles": "CCO"})
            det.append({"molecule_id": mid, "n_candidatos": 0, "top25": ""})
            continue
        top = rank_molecule(best, mols)[:25]
        smiles = []
        for code in top:
            ik = mol_names[code]
            sm = rep.get(ik)
            if sm:
                smiles.append(sm)
        if not smiles:
            smiles = ["CCO"]
        out_rows.append({"molecule_id": mid, "smiles": ";".join(smiles[:25])})
        det.append({"molecule_id": mid, "n_candidatos": int(n_cand),
                    "top25": ";".join(mol_names[c] for c in top)})
        if (ui + 1) % 100 == 0:
            log(f"test: {ui+1:,}/{len(uniq):,} moléculas ({time.time()-t0:.0f}s)")
    sub = pd.DataFrame(out_rows)
    # orden idéntico al sample_submission
    sample = pd.read_csv(DATOS / "sample_submission.csv")
    sub = sample[["molecule_id"]].merge(sub, on="molecule_id", how="left")
    assert len(sub) == 400, len(sub)
    assert sub["smiles"].notna().all()
    n_sm = sub["smiles"].apply(lambda s: len(s.split(";")))
    assert (n_sm >= 1).all() and (n_sm <= 25).all(), n_sm.describe().to_dict()
    assert (n_sm > 1).sum() >= 1
    sub.to_csv(SALIDA / "submission.csv", index=False)
    pd.DataFrame(det).to_csv(VERIF / "test_detalle.csv", index=False)
    log(f"submission escrita: {len(sub)} filas, SMILES por molécula: media {n_sm.mean():.1f}, "
        f"mediana {n_sm.median():.0f}, min {n_sm.min()}, max {n_sm.max()}, "
        f"con 25: {(n_sm==25).sum()}, con 1: {(n_sm==1).sum()}")
    log("PREDICCIÓN COMPLETA (sin envío a Kaggle)")


if __name__ == "__main__":
    main()
