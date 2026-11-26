"""CASMI paso 2 — 05_preparar_entrenamiento.py
Prepara el entrenamiento del modelo lineal espectro -> huella.

- Entrada: val_lib.npz (librería de validación, 2,285,808 espectros de
  261,536 moléculas) + huellas_pool.npz (04_huellas.py).
- Un espectro por molécula de librería: el de mayor nº de picos (nnz),
  determinista. 261,536 filas de entrenamiento.
- Selección de features: top-K bins por frecuencia de documento (df) sobre
  esas filas. K por defecto 50,000 (ver justificación en el informe: la
  matriz de coeficientes 2048 x K en float32 cabe en RAM; K=50k cubre la
  gran mayoría de la masa de picos).
- Transformación de features: sqrt(intensidad) y normalización L2 por fila
  (la misma que usa el coseno del baseline 1).
- Salida: intermedios/X_train_paso2.npz (CSR float32, 261,536 x K),
  y_train_paso2.npz (CSR bool, 261,536 x 2048), train_mol_paso2.npy,
  bins_sel_paso2.npy, verificacion/preparar_paso2.json.
"""
import argparse, time, resource, json
from pathlib import Path
import numpy as np
from scipy import sparse

T0 = time.time()
def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time()-T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)

BASE = Path("/home/hatch/workspace/snt-taxonomia/kaggle-casmi")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=50_000)
    args = ap.parse_args()
    K = args.k

    X = sparse.load_npz(INTER / "val_lib.npz").tocsr()
    meta = np.load(INTER / "val_lib_meta.npz", allow_pickle=True)
    mol = meta["mol"].astype(np.int32)
    log(f"val_lib: {X.shape[0]:,} filas, {X.nnz:,} nnz, {len(np.unique(mol)):,} moléculas")

    # 1 espectro por molécula: el de mayor nnz
    nnz = np.diff(X.indptr).astype(np.int32)
    order = np.lexsort((-nnz, mol))          # primario mol asc, secundario nnz desc
    mol_s = mol[order]
    _, first = np.unique(mol_s, return_index=True)
    sel_rows = order[first]
    train_mol = mol[sel_rows]
    log(f"1 espectro/molécula: {len(sel_rows):,} filas (nnz medio {nnz[sel_rows].mean():.1f})")
    Xs = X[sel_rows]
    del X, order, mol_s
    log(f"X_sel en RAM: {Xs.shape[0]:,} x {Xs.shape[1]:,}, {Xs.nnz:,} nnz")

    # frecuencia de documento y cobertura por K
    df = np.asarray((Xs > 0).sum(axis=0)).ravel().astype(np.int64)
    df_sorted = np.sort(df)[::-1]
    total_nnz = int(Xs.nnz)
    cov = {}
    for k in (10_000, 20_000, 50_000, 100_000, 200_000):
        kk = min(k, len(df_sorted))
        cov[k] = float(df_sorted[:kk].sum() / total_nnz)
    log(f"bins con df>0: {(df>0).sum():,}; df>=10: {(df>=10).sum():,}; "
        f"df>=100: {(df>=100).sum():,}; df>=1000: {(df>=1000).sum():,}")
    log(f"cobertura de nnz por K: " + ", ".join(f"{k}: {v:.3f}" for k, v in cov.items()))

    bins_sel = np.argsort(-df, kind="stable")[:K].astype(np.int32)
    bins_sel.sort()
    df_min = int(df[bins_sel].min())
    log(f"K={K:,}: df mínimo incluido {df_min}, cobertura nnz {cov.get(K, float(df_sorted[:K].sum()/total_nnz)):.3f}")

    # transformación: sqrt + L2 por fila, solo columnas seleccionadas
    Xf = Xs[:, bins_sel].tocsr()
    del Xs
    Xf.data = np.sqrt(Xf.data.astype(np.float32)).astype(np.float32)
    norms = np.sqrt(np.asarray(Xf.sum(axis=1)).ravel().astype(np.float32))
    norms[norms <= 0] = 1.0
    Xf = sparse.diags((1.0 / norms).astype(np.float32)) @ Xf
    Xf = Xf.tocsr().astype(np.float32)
    log(f"X_train: {Xf.shape[0]:,} x {Xf.shape[1]:,}, {Xf.nnz:,} nnz")

    # objetivos: huellas de las moléculas de entrenamiento
    Y = sparse.load_npz(INTER / "huellas_pool.npz").tocsr()
    ytr = Y[train_mol].tocsr()
    n_sin_fp = int((np.diff(ytr.indptr) == 0).sum())
    log(f"y_train: {ytr.shape[0]:,} x {ytr.shape[1]:,}, {ytr.nnz:,} nnz, "
        f"moléculas sin huella: {n_sin_fp} (se quedan con objetivo cero)")
    del Y

    sparse.save_npz(INTER / "X_train_paso2.npz", Xf)
    sparse.save_npz(INTER / "y_train_paso2.npz", ytr)
    np.save(INTER / "train_mol_paso2.npy", train_mol)
    np.save(INTER / "bins_sel_paso2.npy", bins_sel)
    stats = {
        "n_train": int(Xf.shape[0]), "K": int(K), "df_min_incluido": df_min,
        "cobertura_nnz": cov, "nnz_X_train": int(Xf.nnz),
        "n_sin_huella_objetivo": n_sin_fp,
        "bins_df_gt0": int((df > 0).sum()), "bins_df_ge10": int((df >= 10).sum()),
        "bins_df_ge100": int((df >= 100).sum()), "bins_df_ge1000": int((df >= 1000).sum()),
    }
    (VERIF / "preparar_paso2.json").write_text(json.dumps(stats, indent=2))
    log(f"FIN preparar: {stats}")

if __name__ == "__main__":
    main()
