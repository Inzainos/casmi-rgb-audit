"""CASMI paso 2 — 09_predecir_test_paso2.py
Predice huellas para las 400 moléculas de test con un modelo (topo,
lineal o knn), recupera los top-25 SMILES del pool de train por Tanimoto
y escribe salida/submission_paso2.csv. NO toca salida/submission.csv
(baseline 1): verifica al final que su md5 siga siendo
7a7da389417fb162039a94394d76e1ee.

Agregación por molécula: media de las huellas predichas de sus espectros
(documentado; para el modelo lineal equivale a predecir el espectro medio).

También calcula la cobertura fuerte de la búsqueda en librería (baseline 1)
en el test: top-1 coseno por molécula (misma agregación max que el
baseline 1) y el solape del top-25 con salida/submission.csv.

Uso: python3 09_predecir_test_paso2.py --modelo topo|lineal|knn
"""
import argparse, time, json, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
sys.path.insert(0, str(Path(__file__).parent))
from casmi_lib import log, load_library, row_stats, score_query, rank_molecule
from paso2_lib import load_pool, retrieval
from topo_lib import cargar_baseline, predecir_huellas
from importlib import import_module
_construir = import_module("01_construir")
_m06 = import_module("06_entrenar_lineal")

BASE = Path("/home/hatch/workspace/snt-taxonomia/kaggle-casmi")
DATOS = BASE / "datos"
INTER = BASE / "intermedios"
SALIDA = BASE / "salida"
VERIF = BASE / "verificacion"
MD5_BASELINE1 = "7a7da389417fb162039a94394d76e1ee"


def md5_archivo(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def binear_test():
    """Binea test.parquet (top-100, bins 0.01 Da). Devuelve Q CSR, ids, pmz."""
    cache = INTER / "test_q_paso2.npz"
    metac = INTER / "test_q_paso2_meta.npz"
    if cache.exists() and metac.exists():
        Q = sparse.load_npz(cache).tocsr()
        mm = np.load(metac, allow_pickle=True)
        log(f"test bineado cargado de disco: {Q.shape[0]:,} espectros")
        return Q, mm["molecule_id"], mm["precursor_mz"].astype(np.float64)
    test = pd.read_parquet(DATOS / "test.parquet")
    rows, cols, data = [], [], []
    q_id, q_pmz = [], []
    n_drop = 0
    for i, r in test.iterrows():
        pmz = float(r["precursor_mz"])
        b = _construir.binear_fila(r["ms2_mzs"], r["ms2_normalized_intensities"])
        if b is None or not (100.0 <= pmz <= 2000.0):
            n_drop += 1
            continue
        bins, vals = b
        k = len(q_id)
        rows.extend([k] * len(bins))
        cols.extend(bins.tolist())
        data.extend(vals.tolist())
        q_id.append(r["molecule_id"])
        q_pmz.append(pmz)
    Q = sparse.csr_matrix((np.asarray(data, dtype=np.float32),
                           (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
                          shape=(len(q_id), 200_000))
    Q.sum_duplicates()
    sparse.save_npz(cache, Q)
    np.savez(metac, molecule_id=np.asarray(q_id, dtype=object),
             precursor_mz=np.asarray(q_pmz, dtype=np.float64))
    log(f"test bineado: {Q.shape[0]:,} espectros, {Q.nnz:,} nnz, descartados {n_drop}")
    return Q, np.asarray(q_id, dtype=object), np.asarray(q_pmz, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["topo", "lineal", "knn"], required=True)
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--topk_rec", type=int, default=40, help="candidatos a recuperar antes de deduplicar SMILES")
    args = ap.parse_args()

    md5_antes = md5_archivo(SALIDA / "submission.csv")
    log(f"md5 submission.csv (baseline 1) antes: {md5_antes} "
        f"(esperado {MD5_BASELINE1}, coincide: {md5_antes == MD5_BASELINE1})")

    Q, q_id, q_pmz = binear_test()
    pool, pool_nbits = load_pool(INTER)
    mapa = pd.read_parquet(INTER / "mapa_smiles_full.parquet")
    mol_names = np.load(VERIF / "mol_index.npy", allow_pickle=True)
    rep = dict(zip(mapa["inchikey14"], mapa["smiles_rep"]))
    log(f"mapa SMILES: {len(rep):,} moléculas")

    # ---- predicción de huellas por molécula de test ----
    if args.modelo == "topo":
        baseline = cargar_baseline(INTER)
        bins_sel = np.load(INTER / "bins_sel_paso2.npy")
        P_spec, diag_test = predecir_huellas(Q, bins_sel, baseline)
        log(f"P test topo por espectro: {P_spec.shape}, pred media {P_spec.mean():.4f}, "
            f"L2/espectro {diag_test['n_present_L2'].mean():.1f}, "
            f"L1/espectro {diag_test['n_present_L1'].mean():.1f}, "
            f"capturas/espectro {diag_test['n_capture'].mean():.1f}")
        np.savez(INTER / "diag_test_topo.npz", **diag_test)
    elif args.modelo == "lineal":
        W = np.load(INTER / "W_lineal_paso2.npy")
        bins_sel = np.load(INTER / "bins_sel_paso2.npy")
        Qf = _m06.transformar(Q, bins_sel)
        P_spec = np.asarray(Qf @ W, dtype=np.float32)
        del Qf, W
        log(f"P test lineal por espectro: {P_spec.shape}")
    else:
        X, prec, mol_sorted, _ = load_library(INTER, "full_lib")
        rowsum, S_row, norm_cos = row_stats(X)
        pool_bool = sparse.load_npz(INTER / "huellas_pool.npz").tocsr()
        P_spec = np.zeros((Q.shape[0], 2048), dtype=np.float32)
        n_sin = 0
        for j in range(Q.shape[0]):
            m, cos, _en = score_query(X, prec, mol_sorted, rowsum, S_row, norm_cos,
                                      Q[j], float(q_pmz[j]), "ppm")
            if m is None:
                n_sin += 1
                continue
            best = {}
            for code, c in zip(m.tolist(), cos.tolist()):
                if c > best.get(code, -1.0):
                    best[code] = float(c)
            top = sorted(best.items(), key=lambda kv: -kv[1])[:10]
            w = np.clip(np.asarray([c for _, c in top], dtype=np.float32), 0.0, None)
            if w.sum() <= 0:
                continue
            w /= w.sum()
            codes = np.asarray([c for c, _ in top], dtype=np.int32)
            F = pool_bool[codes].toarray().astype(np.float32)
            P_spec[j] = w @ F
        log(f"P test knn por espectro: {P_spec.shape}, espectros sin candidatos: {n_sin}")
        del X, pool_bool

    # agregar por molecule_id (media de espectros)
    order = np.argsort(q_id, kind="stable")
    q_s = q_id[order]
    uniq, starts = np.unique(q_s, return_index=True)
    bounds = list(starts) + [len(q_s)]
    counts = np.diff(bounds).astype(np.float32)
    rows_a = np.repeat(np.arange(len(uniq)), counts.astype(np.int64))
    cols_a = order.astype(np.int32)
    data_a = np.repeat((1.0 / counts).astype(np.float32), counts.astype(np.int64))
    A = sparse.csr_matrix((data_a, (rows_a, cols_a)), shape=(len(uniq), len(q_id)))
    P_test = np.asarray(A @ P_spec, dtype=np.float32)
    del P_spec, A
    log(f"P test por molécula: {P_test.shape}, espectros/mol media {counts.mean():.2f}")

    # ---- recuperación del pool ----
    top_idx, top_sim, _ = retrieval(P_test, pool, pool_nbits, topk=args.topk_rec)
    np.save(INTER / f"P_test_{args.modelo}.npy", P_test)
    np.save(INTER / f"top_idx_test_{args.modelo}.npy", top_idx)

    # ---- construir submission ----
    out_rows = []
    for i, mid in enumerate(uniq):
        smiles = []
        vistos = set()
        for code in top_idx[i]:
            ik = mol_names[code]
            sm = rep.get(ik)
            if sm and sm not in vistos:
                vistos.add(sm)
                smiles.append(sm)
            if len(smiles) >= args.topk:
                break
        out_rows.append({"molecule_id": mid, "smiles": ";".join(smiles)})
    sub = pd.DataFrame(out_rows)
    sample = pd.read_csv(DATOS / "sample_submission.csv")
    sub = sample[["molecule_id"]].merge(sub, on="molecule_id", how="left")
    assert len(sub) == 400, len(sub)
    assert sub["smiles"].notna().all()
    assert (sub["smiles"] != "").all()
    assert (sub["smiles"] != "CCO").all(), "fallback CCO presente"
    n_sm = sub["smiles"].apply(lambda s: len(s.split(";")))
    assert (n_sm >= 1).all() and (n_sm <= 25).all(), n_sm.describe().to_dict()
    assert all(len(set(s.split(";"))) == len(s.split(";")) for s in sub["smiles"])
    sub.to_csv(SALIDA / "submission_paso2.csv", index=False)
    log(f"submission_paso2.csv ({args.modelo}): {len(sub)} filas, SMILES/mol media "
        f"{n_sm.mean():.1f}, mediana {n_sm.median():.0f}, min {n_sm.min()}, max {n_sm.max()}, "
        f"con 25: {(n_sm==25).sum()}")

    # ---- cobertura de la búsqueda en librería (baseline 1) ----
    X, prec, mol_sorted, _ = load_library(INTER, "full_lib")
    rowsum, S_row, norm_cos = row_stats(X)
    cob = []
    for ui, mid in enumerate(uniq):
        s, e = bounds[ui], bounds[ui + 1]
        best, mols = [], []
        n_cand = 0
        for j in order[s:e]:
            m, c, _en = score_query(X, prec, mol_sorted, rowsum, S_row, norm_cos,
                                    Q[j], float(q_pmz[j]), "ppm")
            if m is None:
                continue
            n_cand += len(m)
            best.append(c); mols.append(m)
        top = rank_molecule(best, mols)[:25] if best else []
        top1_cos = float(max([c.max() for c in best])) if best else 0.0
        cob.append({"molecule_id": mid, "top1_cos": top1_cos,
                    "top1_ik": mol_names[top[0]] if top else "",
                    "n_candidatos": int(n_cand),
                    "n_top25": len(top)})
    cob_df = pd.DataFrame(cob)
    cob_df.to_csv(VERIF / "test_cobertura_paso2.csv", index=False)
    # solape con baseline 1
    det = pd.read_csv(VERIF / "test_detalle.csv")
    det_idx = {r.molecule_id: set(r.top25.split(";")) if isinstance(r.top25, str) and r.top25 else set()
               for r in det.itertuples()}
    solapes = []
    for i, mid in enumerate(uniq):
        iks = {mol_names[c] for c in top_idx[i][:25]}
        base = det_idx.get(mid, set())
        solapes.append(len(iks & base))
    solapes = np.asarray(solapes)
    stats = {
        "modelo": args.modelo,
        "n_test": int(len(uniq)),
        "smiles_por_mol_media": float(n_sm.mean()),
        "con_25": int((n_sm == 25).sum()),
        "top1_cos_media": float(cob_df["top1_cos"].mean()),
        "top1_cos_mediana": float(cob_df["top1_cos"].median()),
        "fuerte_ge_0.9": int((cob_df["top1_cos"] >= 0.9).sum()),
        "fuerte_ge_0.8": int((cob_df["top1_cos"] >= 0.8).sum()),
        "fuerte_ge_0.7": int((cob_df["top1_cos"] >= 0.7).sum()),
        "fuerte_ge_0.95": int((cob_df["top1_cos"] >= 0.95).sum()),
        "solape_top25_media": float(solapes.mean()),
        "solape_top25_mediana": float(np.median(solapes)),
        "solape_top25_cero": int((solapes == 0).sum()),
        "md5_submission_baseline1_antes": md5_antes,
    }
    (VERIF / f"test_paso2_{args.modelo}.json").write_text(json.dumps(stats, indent=2))
    log(f"cobertura librería: top1 cos media {stats['top1_cos_media']:.3f}, "
        f"fuerte ≥0.9: {stats['fuerte_ge_0.9']}/400, ≥0.8: {stats['fuerte_ge_0.8']}, "
        f"≥0.7: {stats['fuerte_ge_0.7']}")
    log(f"solape top-25 paso2 vs baseline 1: media {solapes.mean():.1f}, "
        f"mediana {np.median(solapes):.0f}, cero: {(solapes==0).sum()}/400")

    md5_despues = md5_archivo(SALIDA / "submission.csv")
    log(f"md5 submission.csv (baseline 1) después: {md5_despues} "
        f"(intacto: {md5_despues == MD5_BASELINE1})")
    assert md5_despues == MD5_BASELINE1, "¡submission.csv del baseline 1 CAMBIÓ!"
    log(f"FIN predecir test ({args.modelo})")

if __name__ == "__main__":
    main()
