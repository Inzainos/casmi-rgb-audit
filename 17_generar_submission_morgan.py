"""CASMI paso 3 — 17_generar_submission_morgan.py
Genera salida/submission_morgan.csv (400 filas x 25 SMILES) con el mejor
modelo Morgan medido: lineal K=50k (W_lineal_paso2.npy) + variante
lineal_bin (clip [0,1] + top-100 bits a 1) + retrieval con filtro de
masa de precursor (mismo protocolo que 13: candidatos = moléculas con
algún espectro en full_lib dentro de ±10 ppm, piso 0.002 Da, de
cualquiera de los precursores de la consulta; ranking por Tanimoto).

Mol-split con filtro de masa (medido, 13): lineal_bin MRR@25 0.4146
(mejor modelo Morgan); lineal crudo 0.3719; knn 0.1406; oráculo 0.9937.
El binarizado WPC (bin005_full_bin, K=20k) llegó a 0.4023: NO supera
al lineal_bin K=50k, así que el submission usa lineal_bin K=50k.

Reglas: NO toca los tres submissions existentes (md5 antes/después);
el nuevo se llama submission_morgan.csv. SMILES únicos por molécula;
si tras dedup hay <25, se rellena repitiendo el top-1 (regla de
11_rellenar_25.py). Orden de filas = sample_submission.csv.
Sin baseline-1 ni blend: el blend es decisión de Elán (vía paralela).

Uso: python3 17_generar_submission_morgan.py
Salida: salida/submission_morgan.csv, intermedios/P_test_lineal.npy,
  intermedios/top_idx_test_morgan.npy, verificacion/submission_morgan.json
"""
import argparse, time, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import sys
sys.path.insert(0, str(Path(__file__).parent))
from casmi_lib import log
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ev13", Path(__file__).parent / "13_evaluar_masa.py")
_ev13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev13)
tanimoto_vec, topk_de = _ev13.tanimoto_vec, _ev13.topk_de

BASE = Path("/home/hatch/workspace/enveda-casmi-2026")
DATOS = BASE / "datos"
INTER = BASE / "intermedios"
SALIDA = BASE / "salida"
VERIF = BASE / "verificacion"
TOPK = 25
TOPK_REC = 40
MD5_ESPERADOS = {
    "submission.csv": "7a7da389417fb162039a94394d76e1ee",
    "submission_paso2.csv": "ed758b99cdb3a91ac690f02fc3618779",
    "submission_relleno25.csv": "7f082916a4a2bd68392750ea3cf2c8a7",
}

def md5_archivo(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def transformar(Q, bins_sel):
    """sqrt + L2 por fila (copia de 06_entrenar_lineal.transformar)."""
    Qf = Q[:, bins_sel].tocsr()
    Qf.data = np.sqrt(Qf.data.astype(np.float32)).astype(np.float32)
    norms = np.sqrt(np.asarray(Qf.sum(axis=1)).ravel().astype(np.float32))
    norms[norms <= 0] = 1.0
    Qf = sparse.diags((1.0 / norms).astype(np.float32)) @ Qf
    return Qf.tocsr().astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol-ppm", type=float, default=10.0)
    ap.add_argument("--tol-min", type=float, default=0.002)
    args = ap.parse_args()

    for f, esperado in MD5_ESPERADOS.items():
        got = md5_archivo(SALIDA / f)
        log(f"md5 {f} antes: {got} (esperado {esperado}, coincide: {got == esperado})")
        assert got == esperado, f"¡{f} cambió antes de empezar!"

    # ---- 1. predicción lineal para test ----
    Q = sparse.load_npz(INTER / "test_q_paso2.npz").tocsr()
    mm = np.load(INTER / "test_q_paso2_meta.npz", allow_pickle=True)
    q_id = np.asarray(mm["molecule_id"], dtype=object)
    log(f"test bineado: {Q.shape[0]:,} espectros, {len(np.unique(q_id)):,} moléculas")
    W = np.load(INTER / "W_lineal_paso2.npy")
    bins_sel = np.load(INTER / "bins_sel_paso2.npy")
    log(f"W lineal: {W.shape} ({W.nbytes/1e6:.0f} MB), bins_sel {len(bins_sel):,}")
    Qf = transformar(Q, bins_sel)
    del Q
    P_spec = np.asarray(Qf @ W, dtype=np.float32)
    del Qf, W
    log(f"P test lineal por espectro: {P_spec.shape}, media {P_spec.mean():.4f} max {P_spec.max():.3f}")
    # agregar por molecule_id (media)
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
    np.save(INTER / "P_test_lineal.npy", P_test)
    np.save(INTER / "P_test_mol_ids.npy", uniq)
    log(f"P test por molécula: {P_test.shape}, espectros/mol media {counts.mean():.2f}")

    # ---- 2. variante lineal_bin ----
    P_bin = np.zeros_like(P_test)
    for i in range(P_test.shape[0]):
        p = np.clip(P_test[i], 0.0, 1.0)
        P_bin[i, np.argpartition(-p, 100)[:100]] = 1.0
    log("lineal_bin aplicado: clip [0,1] + top-100 bits a 1")

    # ---- 3. retrieval con filtro de masa ----
    pool = sparse.load_npz(INTER / "huellas_pool.npz").tocsr()
    pool_nbits = np.diff(pool.indptr).astype(np.float32)
    idx = np.load(INTER / "precursores_full_idx.npz")
    prec_sorted, mol_sorted = idx["prec"], idx["mol"]
    qp = np.load(INTER / "query_prec_test.npz", allow_pickle=True)
    qp_ids = np.asarray(qp["molecule_id"], dtype=object)
    q_prec, q_indptr = qp["prec"], qp["indptr"]
    assert len(qp_ids) == 400, len(qp_ids)
    # mapa inchikey14 -> SMILES representativo
    mapa = pd.read_parquet(INTER / "mapa_smiles_full.parquet")
    mol_names = np.load(VERIF / "mol_index.npy", allow_pickle=True)
    rep = dict(zip(mapa["inchikey14"], mapa["smiles_rep"]))
    log(f"pool {pool.shape[0]:,} moléculas, mapa SMILES {len(rep):,}, "
        f"query_prec_test {len(qp_ids)} moléculas")
    # índice molecule_id -> fila en P_test (uniq está ordenado; qp_ids no)
    pos_en_P = {mid: i for i, mid in enumerate(uniq)}
    assert set(qp_ids.tolist()) == set(uniq.tolist()), "molecule_ids no coinciden"

    top_idx = np.zeros((400, TOPK_REC), dtype=np.int32)
    n_top = np.zeros(400, dtype=np.int32)
    cand_n = np.zeros(400, dtype=np.int32)
    n_sin_smiles = 0
    t0 = time.time()
    for j, mid in enumerate(qp_ids):
        cands = set()
        for q in q_prec[q_indptr[j]:q_indptr[j + 1]]:
            tol = max(float(q) * args.tol_ppm * 1e-6, args.tol_min)
            lo = int(np.searchsorted(prec_sorted, q - tol))
            hi = int(np.searchsorted(prec_sorted, q + tol))
            if hi > lo:
                cands.update(mol_sorted[lo:hi].tolist())
        codes = np.fromiter(cands, dtype=np.int32)
        cand_n[j] = len(codes)
        if len(codes) == 0:
            continue
        F = pool[codes].toarray().astype(np.float32)
        nb = pool_nbits[codes]
        p = P_bin[pos_en_P[mid]]
        t = tanimoto_vec(p, F, nb)
        top_c, _ = topk_de(t, codes, k=TOPK_REC)
        top_idx[j, :len(top_c)] = top_c
        n_top[j] = len(top_c)
        if (j + 1) % 100 == 0:
            log(f"  retrieval test: {j+1}/400 ({time.time()-t0:.0f}s, "
                f"cand mediana {np.median(cand_n[:j+1]):.0f})")
    np.save(INTER / "top_idx_test_morgan.npy", top_idx)
    np.save(INTER / "n_top_test_morgan.npy", n_top)
    log(f"retrieval test completo: 400 moléculas en {time.time()-t0:.0f}s, "
        f"candidatos/mol mediana {np.median(cand_n):.0f} media {cand_n.mean():.1f} "
        f"min {cand_n.min()} max {cand_n.max()}")

    # ---- 4. construir submission (25 SMILES únicos, relleno top-1) ----
    out_rows = []
    n_relleno = 0
    n_filas_cortas = 0
    for j, mid in enumerate(qp_ids):
        smiles = []
        vistos = set()
        for code in top_idx[j, :n_top[j]]:
            ik = mol_names[code]
            sm = rep.get(ik)
            if not sm:
                n_sin_smiles += 1
                continue
            if sm not in vistos:
                vistos.add(sm)
                smiles.append(sm)
            if len(smiles) >= TOPK:
                break
        if len(smiles) == 0:
            raise RuntimeError(f"molécula {mid} sin ningún SMILES candidato")
        if len(smiles) < TOPK:
            n_filas_cortas += 1
            n_relleno += TOPK - len(smiles)
            smiles += [smiles[0]] * (TOPK - len(smiles))
        out_rows.append({"molecule_id": mid, "smiles": ";".join(smiles)})
    sub = pd.DataFrame(out_rows)
    sample = pd.read_csv(DATOS / "sample_submission.csv")
    sub = sample[["molecule_id"]].merge(sub, on="molecule_id", how="left")
    assert len(sub) == 400, len(sub)
    assert sub["smiles"].notna().all()
    assert (sub["smiles"] != "").all()
    assert (sub["smiles"] != "CCO").all(), "fallback CCO presente"
    n_sm = sub["smiles"].apply(lambda s: len(s.split(";")))
    assert (n_sm == 25).all(), n_sm.describe().to_dict()
    sub.to_csv(SALIDA / "submission_morgan.csv", index=False)
    log(f"submission_morgan.csv: {len(sub)} filas, 25 SMILES c/u, "
        f"filas que necesitaron relleno: {n_filas_cortas} (+{n_relleno} repeticiones), "
        f"códigos sin SMILES saltados: {n_sin_smiles}")

    # solape con baseline 1 (por SMILES)
    base = pd.read_csv(SALIDA / "submission.csv")
    base_sets = {r.molecule_id: set(r.smiles.split(";")) for r in base.itertuples()}
    solapes = np.array([len(set(s.split(";")) & base_sets[m]) for s, m in
                        zip(sub["smiles"], sub["molecule_id"])])
    stats = {
        "modelo": "lineal K=50k + lineal_bin + filtro masa ±10ppm",
        "n_test": 400, "smiles_por_mol": 25,
        "cand_mediana": float(np.median(cand_n)), "cand_media": float(cand_n.mean()),
        "cand_min": int(cand_n.min()), "cand_max": int(cand_n.max()),
        "filas_con_relleno": int(n_filas_cortas),
        "repeticiones_relleno": int(n_relleno),
        "codigos_sin_smiles_saltados": int(n_sin_smiles),
        "solape_top25_vs_baseline1_media": float(solapes.mean()),
        "solape_top25_vs_baseline1_mediana": float(np.median(solapes)),
        "solape_top25_vs_baseline1_cero": int((solapes == 0).sum()),
        "solape_top25_vs_baseline1_25": int((solapes == 25).sum()),
    }
    (VERIF / "submission_morgan.json").write_text(json.dumps(stats, indent=2))
    log(f"solape top-25 Morgan vs baseline 1: media {solapes.mean():.1f}, "
        f"mediana {np.median(solapes):.0f}, cero: {(solapes==0).sum()}/400, "
        f"iguales (25): {(solapes==25).sum()}/400")

    for f, esperado in MD5_ESPERADOS.items():
        got = md5_archivo(SALIDA / f)
        log(f"md5 {f} después: {got} (intacto: {got == esperado})")
        assert got == esperado, f"¡{f} CAMBIÓ!"
    log("FIN generar submission_morgan.csv")

if __name__ == "__main__":
    main()
