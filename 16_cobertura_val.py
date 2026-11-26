"""CASMI paso 3 — 16_cobertura_val.py
Mide el top-1 coseno de la búsqueda en librería (baseline 1) para las
moléculas de validación, EXCLUYENDO la propia molécula de la librería
en mol-split (la librería representativa X_train_paso2.npz ya excluye
las moléculas mol-split por construcción: 1 espectro por molécula,
el de mayor nnz, sqrt+L2 sobre los 50k bins de 05).

Es la pieza que falta para validar el blend "top-1 coseno >= 0.95 ->
baseline 1, si no -> Morgan": en mol-split (moléculas nuevas) un
top-1 >= 0.95 sería un enrutamiento FALSO (la molécula verdadera no
está en la librería); en spec-split (en librería) debería ser ~1.0
por construcción. Agregación max por molécula, igual que
test_cobertura_paso2.csv (misma que el baseline 1).

HALLAZGO DE NORMALIZACIÓN (medido aquí): la "L2" de 05/06 NO es la
norma L2 verdadera: usa norms = sqrt(Σ sqrt(int)) en vez de
sqrt(Σ int), así que los vectores de X_train y de las consultas NO
son unitarios (norma media ~0.5-0.6). El dot directo NO es un coseno
y NO se puede comparar con el umbral 0.95 (que viene del coseno
verdadero de 03/09, donde top1_cos = 1.0 para las 400 de test).
Por eso este script RENORMALIZA cada vector a norma L2 verdadera
antes del dot: el top-1 reportado SÍ es el coseno verdadero.
Los resultados Morgan (05/06/13/15/17) no se ven afectados: usan la
misma transformación en entrenar y predecir, así que sus MRR son
válidos como medidos; la normalización es solo un escalado de
features, consistente en todo el camino.

Aproximación declarada: el top-1 se calcula contra el espectro
representativo (max-nnz) de cada molécula candidata, no contra todos
sus espectros; subestima levemente el top-1 real (el representativo
es el más informativo, así que la subestimación es pequeña).

Uso: python3 16_cobertura_val.py
Salida: verificacion/top1cos_val_mol.npy, top1cos_val_spec.npy,
  verificacion/cobertura_val.json
"""
import time, json
from pathlib import Path
import numpy as np
from scipy import sparse
import sys
sys.path.insert(0, str(Path(__file__).parent))
from casmi_lib import log

BASE = Path("/home/hatch/workspace/enveda-casmi-2026")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"

def transformar(Q, bins_sel):
    """sqrt + L2 por fila (copia de 06_entrenar_lineal.transformar)."""
    Qf = Q[:, bins_sel].tocsr()
    Qf.data = np.sqrt(Qf.data.astype(np.float32)).astype(np.float32)
    norms = np.sqrt(np.asarray(Qf.sum(axis=1)).ravel().astype(np.float32))
    norms[norms <= 0] = 1.0
    Qf = sparse.diags((1.0 / norms).astype(np.float32)) @ Qf
    return Qf.tocsr().astype(np.float32)

def l2_verdadera(X):
    """Renormaliza cada fila CSR a norma L2 verdadera (=1). Devuelve
    (X_norm, normas_originales)."""
    norms = np.sqrt(np.asarray(X.power(2).sum(axis=1)).ravel().astype(np.float32))
    safe = norms.copy()
    safe[safe <= 0] = 1.0
    Xn = sparse.diags((1.0 / safe).astype(np.float32)) @ X
    return Xn.tocsr().astype(np.float32), norms

def main():
    X = sparse.load_npz(INTER / "X_train_paso2.npz").tocsr()
    train_mol = np.load(INTER / "train_mol_paso2.npy").astype(np.int32)
    bins_sel = np.load(INTER / "bins_sel_paso2.npy")
    log(f"librería representativa: {X.shape[0]:,} moléculas x {X.shape[1]:,} bins, "
        f"nnz {X.nnz:,}")
    # norma L2 verdadera de las filas de X_train (diagnóstico de 05)
    _, x_norms = l2_verdadera(X)
    log(f"norma L2 verdadera de X_train: media {x_norms.mean():.4f} "
        f"mediana {np.median(x_norms):.4f} (si fuera 1.0, la 'L2' de 05 sería exacta)")
    X, _ = l2_verdadera(X)
    pos = np.full(275_810, -1, dtype=np.int64)
    pos[train_mol] = np.arange(len(train_mol))
    idx = np.load(INTER / "precursores_full_idx.npz")
    prec_sorted, mol_sorted = idx["prec"], idx["mol"]

    out = {}
    for split, ptag, qtag in (("mol", "val", "val_mol"), ("spec", "spec", "val_spec")):
        Q = sparse.load_npz(INTER / f"{ptag}_qmol.npz" if split == "mol"
                            else INTER / "val_qspec.npz").tocsr()
        m = np.load(INTER / f"{ptag}_qmol_meta.npz" if split == "mol"
                    else INTER / "val_qspec_meta.npz", allow_pickle=True)
        mol_q = m["mol"].astype(np.int32)
        qp = np.load(INTER / f"query_prec_{qtag}.npz")
        q_mol, q_prec, q_indptr = qp["mol"].astype(np.int32), qp["prec"], qp["indptr"]
        assert np.array_equal(np.unique(mol_q), q_mol) or len(q_mol) == len(np.unique(mol_q)), \
            "códigos de consulta no coinciden"
        Qf = transformar(Q, bins_sel)
        del Q
        Qf, q_norms = l2_verdadera(Qf)
        log(f"  norma L2 verdadera de las consultas {split} (tras transformar 06): "
            f"media {q_norms.mean():.4f} mediana {np.median(q_norms):.4f}")
        # filas de Qf por molécula (orden de q_mol = orden de uniq(mol_q))
        order = np.argsort(mol_q, kind="stable")
        uniq, starts = np.unique(mol_q[order], return_index=True)
        assert np.array_equal(uniq, q_mol), "orden de moléculas no coincide"
        bounds = list(starts) + [len(order)]
        top1 = np.zeros(len(q_mol), dtype=np.float32)
        t0 = time.time()
        for i in range(len(q_mol)):
            cands = set()
            for q in q_prec[q_indptr[i]:q_indptr[i + 1]]:
                tol = max(float(q) * 10.0 * 1e-6, 0.002)
                lo = int(np.searchsorted(prec_sorted, q - tol))
                hi = int(np.searchsorted(prec_sorted, q + tol))
                if hi > lo:
                    cands.update(mol_sorted[lo:hi].tolist())
            rows = np.array([pos[c] for c in cands if pos[c] >= 0], dtype=np.int64)
            if len(rows) == 0:
                continue
            best = 0.0
            for j in order[starts[i]:bounds[i + 1] if i + 1 < len(starts) else len(order)]:
                qv = np.asarray(Qf[j].toarray()).ravel().astype(np.float32)
                cos = np.asarray(X[rows] @ qv).ravel()
                if len(cos):
                    best = max(best, float(cos.max()))
            top1[i] = best
            if (i + 1) % 4000 == 0:
                log(f"  cobertura {split}: {i+1:,}/{len(q_mol):,} ({time.time()-t0:.0f}s)")
        np.save(VERIF / f"top1cos_val_{split}.npy", top1)
        d = {"n": int(len(top1)), "media": float(top1.mean()),
             "mediana": float(np.median(top1)),
             "p90": float(np.percentile(top1, 90)),
             "max": float(top1.max()),
             "frac_ge_0.95": float((top1 >= 0.95).mean()),
             "frac_ge_0.90": float((top1 >= 0.90).mean()),
             "frac_ge_0.99": float((top1 >= 0.99).mean()),
             "frac_ge_0.80": float((top1 >= 0.80).mean()),
             "n_ge_0.95": int((top1 >= 0.95).sum())}
        out[split] = d
        out[split]["norma_L2_consultas_media"] = float(q_norms.mean())
        out[split]["norma_L2_consultas_mediana"] = float(np.median(q_norms))
        log(f"cobertura {split}: n={d['n']:,} media {d['media']:.4f} mediana {d['mediana']:.4f} "
            f"p90 {d['p90']:.4f} max {d['max']:.4f} | >=0.95: {d['n_ge_0.95']:,} "
            f"({100*d['frac_ge_0.95']:.2f}%) >=0.90: {100*d['frac_ge_0.90']:.2f}% "
            f">=0.99: {100*d['frac_ge_0.99']:.2f}%")
        del Qf
    out["norma_L2_X_train_media"] = float(x_norms.mean())
    out["norma_L2_X_train_mediana"] = float(np.median(x_norms))
    (VERIF / "cobertura_val.json").write_text(json.dumps(out, indent=2))
    log("FIN cobertura val")

if __name__ == "__main__":
    main()
