"""CASMI ronda 2 — 20_rgb.py
Construye la matriz de entrenamiento RGB (3 canales por bin m/z) para el
lineal espectro -> huella Morgan. Candidato (a) de
wpc-schumann-ideas-cap37.md: el espectro como imagen de 3 canales.

Replica EXACTAMENTE la selección de 05_preparar_entrenamiento.py (1 espectro
por molécula, el de mayor nnz; top-K bins por frecuencia de documento) y lo
verifica: train_mol debe ser idéntico a train_mol_paso2.npy y los K bins
elegidos deben ser subconjunto de bins_sel_paso2.npy.

Canales (misma normalización "L2" de 06/14: dividir por sqrt(suma de valores
de la fila); para el canal binario eso es sqrt(nº de picos)):
- sqrt: sqrt(intensidad)
- log:  log1p(intensidad)
- bin:  (intensidad > 0) -> 1.0
Salida:
- intermedios/X_train_rgb.npz (CSR float32, 261,536 x 3K)
- intermedios/bins_rgb.npy (K bins, orden ascendente)
- verificacion/preparar_rgb.json (stats + verificaciones)
Reglas: nada a Kaggle/GitHub; log en verificacion/; un solo proceso pesado.
"""
import argparse, time, resource, json, sys
from pathlib import Path
import numpy as np
from scipy import sparse

T0 = time.time()
def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time()-T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)

BASE = Path("/home/deamon/wt_deamonx/CASMI-sandbox/pruebas")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"

def l2_por_fila(X):
    """Igual que en 14: dividir por sqrt(suma de valores de la fila)."""
    norms = np.sqrt(np.asarray(X.sum(axis=1)).ravel().astype(np.float32))
    norms[norms <= 0] = 1.0
    X = sparse.diags((1.0 / norms).astype(np.float32)) @ X
    return X.tocsr().astype(np.float32)

def canal(Xb, modo):
    Xc = Xb.copy().tocsr()
    if modo == "sqrt":
        Xc.data = np.sqrt(Xc.data.astype(np.float32)).astype(np.float32)
    elif modo == "log":
        Xc.data = np.log1p(Xc.data.astype(np.float32)).astype(np.float32)
    elif modo == "bin":
        Xc.data = np.ones_like(Xc.data, dtype=np.float32)
    else:
        raise ValueError(modo)
    return l2_por_fila(Xc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-cols", type=int, default=20_000)
    args = ap.parse_args()
    K = args.k_cols
    log(f"preparar RGB: K por canal={K:,}, python {sys.version.split()[0]}, "
        f"numpy {np.__version__}")

    X = sparse.load_npz(INTER / "val_lib.npz").tocsr()
    meta = np.load(INTER / "val_lib_meta.npz", allow_pickle=True)
    mol = meta["mol"].astype(np.int32)
    log(f"val_lib: {X.shape[0]:,} filas, {X.nnz:,} nnz, {len(np.unique(mol)):,} moléculas")

    # 1 espectro por molécula: el de mayor nnz (código idéntico a 05)
    nnz = np.diff(X.indptr).astype(np.int32)
    order = np.lexsort((-nnz, mol))
    mol_s = mol[order]
    _, first = np.unique(mol_s, return_index=True)
    sel_rows = order[first]
    train_mol = mol[sel_rows]
    log(f"1 espectro/molécula: {len(sel_rows):,} filas (nnz medio {nnz[sel_rows].mean():.1f})")

    # verificación 1: mismo orden de filas que 05
    train_mol_ref = np.load(INTER / "train_mol_paso2.npy")
    assert np.array_equal(train_mol, train_mol_ref), \
        "train_mol NO coincide con train_mol_paso2.npy: las filas no alinean con y_train"
    log("verificación 1 OK: train_mol idéntico a train_mol_paso2.npy")

    Xs = X[sel_rows]
    del X, order, mol_s
    df = np.asarray((Xs > 0).sum(axis=0)).ravel().astype(np.int64)
    bins_sel = np.argsort(-df, kind="stable")[:K].astype(np.int32)
    bins_sel.sort()

    # verificación 2: los K bins son subconjunto de los 50k de 05
    bins_ref = np.load(INTER / "bins_sel_paso2.npy")
    assert np.isin(bins_sel, bins_ref).all(), \
        "bins RGB NO son subconjunto de bins_sel_paso2.npy"
    log(f"verificación 2 OK: {K:,} bins son subconjunto de los 50k de 05 "
        f"(df mínimo incluido {int(df[bins_sel].min())})")

    Xb = Xs[:, bins_sel].tocsr()
    del Xs
    log(f"Xb: {Xb.shape[0]:,} x {Xb.shape[1]:,}, nnz {Xb.nnz:,}")

    nnz_canal = {}
    canales = {}
    for modo in ("sqrt", "log", "bin"):
        canales[modo] = canal(Xb, modo)
        nnz_canal[modo] = int(canales[modo].nnz)
        log(f"canal {modo}: nnz {nnz_canal[modo]:,}, "
            f"media por fila {nnz_canal[modo] / canales[modo].shape[0]:.1f}")
    Xrgb = sparse.hstack([canales["sqrt"], canales["log"], canales["bin"]],
                         format="csr").astype(np.float32)
    del canales, Xb
    log(f"X_train_rgb: {Xrgb.shape[0]:,} x {Xrgb.shape[1]:,}, nnz {Xrgb.nnz:,}")
    sparse.save_npz(INTER / "X_train_rgb.npz", Xrgb)
    np.save(INTER / "bins_rgb.npy", bins_sel)
    stats = {"k_por_canal": int(K), "n_cols_total": int(Xrgb.shape[1]),
             "n_train": int(Xrgb.shape[0]), "nnz_total": int(Xrgb.nnz),
             "nnz_por_canal": nnz_canal,
             "df_min_incluido": int(df[bins_sel].min()),
             "verif_train_mol_identico_05": True,
             "verif_bins_subconjunto_50k": True,
             "python": sys.version.split()[0], "numpy": np.__version__}
    (VERIF / "preparar_rgb.json").write_text(json.dumps(stats, indent=2))
    log(f"FIN preparar RGB: {stats}")

if __name__ == "__main__":
    main()
