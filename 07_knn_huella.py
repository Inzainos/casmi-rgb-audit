"""CASMI paso 2 — 07_knn_huella.py
Modelo B (comparación): kNN espectral -> huella.

Para cada espectro de consulta: coseno contra la librería de validación
(ventana ±10 ppm, piso 0.002 Da, igual que el baseline 1), top-10 moléculas
por coseno, huella predicha = media ponderada por coseno de las huellas de
esas moléculas. Luego media por molécula sobre sus espectros.

Es el modelo "los análogos más cercanos comparten subestructuras": si la
molécula de consulta tiene análogos en la librería, su huella se aproxima
bien; si no los tiene, la predicción es pobre. En mol-split la molécula
NO está en la librería, así que mide exactamente eso.

Salida:
- intermedios/P_val_knn.npy + P_val_mol_knn.npy
- intermedios/P_spec_knn.npy + P_spec_mol_knn.npy
- verificacion/knn_paso2.json
"""
import time, resource, json, sys
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).parent))
from casmi_lib import log, load_library, row_stats, score_query
from importlib import import_module
_m06 = import_module("06_entrenar_lineal")

BASE = Path("/home/hatch/workspace/snt-taxonomia/kaggle-casmi")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"
NBITS = 2048
TOPK_VECINOS = 10


def predecir_consultas(X, prec, mol_sorted, rowsum, S_row, norm_cos,
                       Q, mol_q, prec_q, pool_bool):
    """Predice la huella por espectro y agrega por molécula.
    Devuelve (P_mol (n_mol x 2048) float32, códigos, conteos, stats)."""
    n_q = Q.shape[0]
    P_spec = np.zeros((n_q, NBITS), dtype=np.float32)
    n_sin_cand = 0
    top1_cos = []
    t0 = time.time()
    for j in range(n_q):
        m, cos, _ent = score_query(X, prec, mol_sorted, rowsum, S_row, norm_cos,
                                   Q[j], float(prec_q[j]), "ppm")
        if m is None:
            n_sin_cand += 1
            continue
        # mejor coseno por molécula
        best = {}
        for code, c in zip(m.tolist(), cos.tolist()):
            if c > best.get(code, -1.0):
                best[code] = float(c)
        top = sorted(best.items(), key=lambda kv: -kv[1])[:TOPK_VECINOS]
        top1_cos.append(top[0][1])
        w = np.asarray([c for _, c in top], dtype=np.float32)
        w = np.clip(w, 0.0, None)
        if w.sum() <= 0:
            continue
        w /= w.sum()
        codes = np.asarray([c for c, _ in top], dtype=np.int32)
        F = pool_bool[codes].toarray().astype(np.float32)
        P_spec[j] = w @ F
        if (j + 1) % 10_000 == 0:
            log(f"  knn: {j+1:,}/{n_q:,} espectros ({time.time()-t0:.0f}s)")
    Pm, uniq, counts = _m06.agregar_por_mol(P_spec, mol_q)
    stats = {"n_espectros": int(n_q), "n_sin_candidatos": int(n_sin_cand),
             "top1_cos_media": float(np.mean(top1_cos)) if top1_cos else 0.0,
             "top1_cos_mediana": float(np.median(top1_cos)) if top1_cos else 0.0,
             "top1_cos_ge_0.9": float(np.mean(np.asarray(top1_cos) >= 0.9)) if top1_cos else 0.0,
             "n_moleculas": int(len(uniq))}
    return Pm, uniq, counts, stats


if __name__ == "__main__":
    X, prec, mol_sorted, _ = load_library(INTER, "val_lib")
    rowsum, S_row, norm_cos = row_stats(X)
    pool_bool = sparse.load_npz(INTER / "huellas_pool.npz").tocsr()
    log(f"pool bool cargado: {pool_bool.shape}, nnz {pool_bool.nnz:,}")

    out = {}
    for tag, npz, meta in (("val", "val_qmol", "val_qmol_meta"),
                           ("spec", "val_qspec", "val_qspec_meta")):
        Q = sparse.load_npz(INTER / f"{npz}.npz").tocsr()
        mm = np.load(INTER / f"{meta}.npz", allow_pickle=True)
        mol_q = mm["mol"].astype(np.int32)
        prec_q = mm["precursor_mz"].astype(np.float64)
        log(f"knn {tag}: {Q.shape[0]:,} espectros")
        Pm, uniq, counts, st = predecir_consultas(
            X, prec, mol_sorted, rowsum, S_row, norm_cos, Q, mol_q, prec_q, pool_bool)
        np.save(INTER / f"P_{tag}_knn.npy", Pm)
        np.save(INTER / f"P_{tag}_mol_knn.npy", uniq)
        out[tag] = st
        log(f"P_{tag}_knn: {Pm.shape[0]:,} moléculas, pred media {Pm.mean():.4f}, "
            f"max {Pm.max():.3f}, top1 cos media {st['top1_cos_media']:.3f}")
        del Q
    (VERIF / "knn_paso2.json").write_text(json.dumps(out, indent=2))
    log(f"FIN knn: {out}")
