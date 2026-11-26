"""CASMI paso 3 — 18_bandas.py
WPC Idea 2: conteos de picos por banda m/z (el `activity_pct` del espectro).

Por espectro: nº de picos presentes (no nulos) en 20 bandas gruesas de
m/z (0-2000 Da en bandas de 100 Da) + nº total de picos + fracción de
bins ocupados. Análogo directo del activity_pct de Sentinel: un puñado
de números globales que resumen "cuánta actividad" hay y dónde, sin
guardar el espectro entero.

Fuentes (sin re-binear nada):
- Train: X_train_paso2.npz (261,536 x 50,000, sqrt+L2; 1 espectro por
  molécula, el de máx nnz) + bins_sel_paso2.npy (bin m/z de cada columna:
  m/z = bin * 0.01 Da).
- Val: val_qmol.npz (38,377 x 200,000, intensidades crudas; la banda sale
  del índice de columna: m/z = col * 0.01 Da).

Salida:
- intermedios/B_train_paso2.npy (261,536 x 22, float32)
- intermedios/B_val_qmol.npy (38,377 x 22, float32)
- verificacion/bandas_stats.json (medias, rangos, tiempo)

Columnas: 20 conteos por banda + total_picos + frac_ocupados.
Uso: 19_entrenar_lineal_bandas.py entrena el lineal solo con bandas.
Reglas: nada a Kaggle/GitHub; log en verificacion/.
"""
import time, resource, json, sys
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
NBANDAS = 20
ANCHO_BANDA_DA = 100.0

def bandas_desde_csr(X, col_a_mz):
    """Conteos de no-nulos por banda m/z + total + fracción ocupada.
    X: CSR (n x ncols); col_a_mz: m/z de cada columna (ncols,)."""
    n, ncols = X.shape
    banda_col = np.minimum((col_a_mz // ANCHO_BANDA_DA).astype(np.int32), NBANDAS - 1)
    # conteo por (fila, banda) vía bincount sobre pares
    indptr, indices = X.indptr, X.indices
    filas = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
    bandas_nnz = banda_col[indices]
    codigo = filas.astype(np.int64) * NBANDAS + bandas_nnz
    conteo = np.bincount(codigo, minlength=n * NBANDAS).reshape(n, NBANDAS)
    total = np.diff(indptr).astype(np.float32)
    frac = (total / ncols).astype(np.float32)
    B = np.hstack([conteo.astype(np.float32),
                   total[:, None], frac[:, None]]).astype(np.float32)
    return B

def main():
    log(f"entorno: python {sys.version.split()[0]}, numpy {np.__version__}")
    bins_sel = np.load(INTER / "bins_sel_paso2.npy")
    mz_train = bins_sel.astype(np.float32) * 0.01
    X = sparse.load_npz(INTER / "X_train_paso2.npz").tocsr()
    log(f"X_train {X.shape}, nnz {X.nnz:,}")
    B_train = bandas_desde_csr(X, mz_train)
    np.save(INTER / "B_train_paso2.npy", B_train)
    log(f"B_train {B_train.shape} guardado; medias por banda: "
        f"{np.round(B_train[:, :NBANDAS].mean(axis=0), 1).tolist()}")
    log(f"  total picos media {B_train[:, 20].mean():.1f}, "
        f"frac ocupada media {B_train[:, 21].mean():.4f}")
    del X
    Q = sparse.load_npz(INTER / "val_qmol.npz").tocsr()
    mz_val = np.arange(Q.shape[1], dtype=np.float32) * 0.01
    log(f"val_qmol {Q.shape}, nnz {Q.nnz:,}")
    B_val = bandas_desde_csr(Q, mz_val)
    np.save(INTER / "B_val_qmol.npy", B_val)
    log(f"B_val {B_val.shape} guardado; medias por banda: "
        f"{np.round(B_val[:, :NBANDAS].mean(axis=0), 1).tolist()}")
    stats = {"n_train": int(B_train.shape[0]), "n_val": int(B_val.shape[0]),
             "n_features": int(B_train.shape[1]),
             "bandas_train_media": [float(v) for v in B_train[:, :NBANDAS].mean(axis=0)],
             "bandas_val_media": [float(v) for v in B_val[:, :NBANDAS].mean(axis=0)],
             "total_train_media": float(B_train[:, 20].mean()),
             "total_val_media": float(B_val[:, 20].mean()),
             "frac_train_media": float(B_train[:, 21].mean()),
             "frac_val_media": float(B_val[:, 21].mean()),
             "python": sys.version.split()[0], "numpy": np.__version__}
    (VERIF / "bandas_stats.json").write_text(json.dumps(stats, indent=2))
    log("FIN bandas")

if __name__ == "__main__":
    main()
