"""CASMI — 25_preparar_ce.py
Prepara las columnas CE para el experimento CE-como-feature (orden de Elán 2026-11-11 ~12:39).

- Verifica alineamiento de ce_val_lib / ce_val_qmol contra val_lib_meta /
  val_qmol_meta (precursor_mz y mol elemento a elemento).
- Reproduce la selección de 05 (1 espectro/molécula: max nnz) sobre val_lib y
  verifica que train_mol_paso2.npy coincide con mol[sel_rows].
- Construye las columnas CE de entrenamiento (261,536 filas) y validación (38,377):
  first: clip(ce_first,0,100)/100, sin CE -> 0.22 (mediana 22)
  mean:  clip(ce_mean,0,100)/100, sin CE -> 0.40 (mediana 40)
  onehot: 5 buckets por ce_first {20, 40, 60, otro, sin CE}
Salida: intermedios/ce_train_first.npy, ce_train_mean.npy, ce_train_onehot.npy,
        ce_val_first.npy, ce_val_mean.npy, ce_val_onehot.npy
        + verificacion/preparar_ce.json
Reglas: nada a Kaggle/GitHub; log en verificacion/; un solo proceso pesado.
"""
import time, resource, json
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

def col_first(ce):
    """ce: dict con ce_n, ce_first. Devuelve columna float32 clip[0,100]/100, NaN->0.22."""
    v = np.asarray(ce["ce_first"], dtype=np.float32).copy()
    assert np.isnan(v).sum() == int((np.asarray(ce["ce_n"]) == 0).sum()), "NaN debe coincidir con ce_n==0"
    v = np.where(np.isnan(v), 22.0, v)
    v = np.clip(v, 0.0, 100.0) / 100.0
    return v.astype(np.float32)

def col_mean(ce):
    v = np.asarray(ce["ce_mean"], dtype=np.float32).copy()
    assert np.isnan(v).sum() == int((np.asarray(ce["ce_n"]) == 0).sum()), "NaN debe coincidir con ce_n==0"
    v = np.where(np.isnan(v), 40.0, v)
    v = np.clip(v, 0.0, 100.0) / 100.0
    return v.astype(np.float32)

def col_onehot(ce):
    """5 buckets por ce_first: 20, 40, 60, otro, sin CE. Devuelve (n,5) float32."""
    n = np.asarray(ce["ce_n"])
    f = np.asarray(ce["ce_first"], dtype=np.float32)
    oh = np.zeros((len(n), 5), dtype=np.float32)
    sin = n == 0
    oh[sin, 4] = 1.0
    oh[(~sin) & (f == 20.0), 0] = 1.0
    oh[(~sin) & (f == 40.0), 1] = 1.0
    oh[(~sin) & (f == 60.0), 2] = 1.0
    oh[(~sin) & (f != 20.0) & (f != 40.0) & (f != 60.0), 3] = 1.0
    assert np.allclose(oh.sum(axis=1), 1.0), "cada fila debe tener exactamente un bucket"
    return oh

def main():
    # --- 1. alineamiento CE vs metas ---
    verif = {}
    ce_lib = np.load(INTER / "ce_val_lib.npz")
    meta_lib = np.load(INTER / "val_lib_meta.npz", allow_pickle=True)
    d_prec = float(np.abs(np.asarray(ce_lib["precursor_mz"]) - np.asarray(meta_lib["precursor_mz"])).max())
    d_mol = int((np.asarray(ce_lib["mol"]) != np.asarray(meta_lib["mol"])).sum())
    verif["ce_val_lib_vs_meta"] = {"max_dif_precursor_mz": d_prec, "n_mol_distintos": d_mol,
                                    "n": int(len(ce_lib["mol"]))}
    log(f"ce_val_lib vs val_lib_meta: max|d_prec|={d_prec}, mol distintos={d_mol}")
    assert d_prec == 0.0 and d_mol == 0, "alineamiento ce_val_lib FALLA"

    ce_q = np.load(INTER / "ce_val_qmol.npz")
    meta_q = np.load(INTER / "val_qmol_meta.npz", allow_pickle=True)
    d_prec = float(np.abs(np.asarray(ce_q["precursor_mz"]) - np.asarray(meta_q["precursor_mz"])).max())
    d_mol = int((np.asarray(ce_q["mol"]) != np.asarray(meta_q["mol"])).sum())
    verif["ce_val_qmol_vs_meta"] = {"max_dif_precursor_mz": d_prec, "n_mol_distintos": d_mol,
                                     "n": int(len(ce_q["mol"]))}
    log(f"ce_val_qmol vs val_qmol_meta: max|d_prec|={d_prec}, mol distintos={d_mol}")
    assert d_prec == 0.0 and d_mol == 0, "alineamiento ce_val_qmol FALLA"

    # --- 2. reproducir selección 05 (1 espectro/molécula, max nnz) ---
    Xlib = sparse.load_npz(INTER / "val_lib.npz").tocsr()
    mol_lib = np.asarray(meta_lib["mol"]).astype(np.int32)
    nnz = np.diff(Xlib.indptr).astype(np.int32)
    order = np.lexsort((-nnz, mol_lib))
    mol_s = mol_lib[order]
    _, first = np.unique(mol_s, return_index=True)
    sel_rows = order[first]
    del Xlib, order, mol_s
    train_mol = np.load(INTER / "train_mol_paso2.npy")
    coincide = bool(np.array_equal(mol_lib[sel_rows], train_mol))
    verif["seleccion_05_reproducida"] = {"n_sel": int(len(sel_rows)),
                                          "train_mol_coincide": coincide}
    log(f"selección 05: {len(sel_rows):,} filas; train_mol coincide={coincide}")
    assert coincide, "la selección reproducida NO coincide con train_mol_paso2.npy"

    # --- 3. columnas CE de entrenamiento (filas sel_rows de ce_val_lib) ---
    ce_tr = {k: np.asarray(ce_lib[k])[sel_rows] for k in ("ce_n", "ce_first", "ce_mean")}
    f_tr, m_tr, oh_tr = col_first(ce_tr), col_mean(ce_tr), col_onehot(ce_tr)
    np.save(INTER / "ce_train_first.npy", f_tr)
    np.save(INTER / "ce_train_mean.npy", m_tr)
    np.save(INTER / "ce_train_onehot.npy", oh_tr)
    verif["train"] = {"n": int(len(f_tr)),
                       "sin_ce": int((np.asarray(ce_tr["ce_n"]) == 0).sum()),
                       "first_media": float(f_tr.mean()), "mean_media": float(m_tr.mean()),
                       "onehot_por_bucket": [int(x) for x in oh_tr.sum(axis=0)]}
    log(f"train: n={len(f_tr):,}, sin CE={verif['train']['sin_ce']:,}, " +
        f"onehot buckets={verif['train']['onehot_por_bucket']}")

    # --- 4. columnas CE de validación (filas de ce_val_qmol) ---
    ce_v = {k: np.asarray(ce_q[k]) for k in ("ce_n", "ce_first", "ce_mean")}
    f_v, m_v, oh_v = col_first(ce_v), col_mean(ce_v), col_onehot(ce_v)
    np.save(INTER / "ce_val_first.npy", f_v)
    np.save(INTER / "ce_val_mean.npy", m_v)
    np.save(INTER / "ce_val_onehot.npy", oh_v)
    verif["val"] = {"n": int(len(f_v)),
                     "sin_ce": int((np.asarray(ce_v["ce_n"]) == 0).sum()),
                     "first_media": float(f_v.mean()), "mean_media": float(m_v.mean()),
                     "onehot_por_bucket": [int(x) for x in oh_v.sum(axis=0)]}
    log(f"val: n={len(f_v):,}, sin CE={verif['val']['sin_ce']:,}, " +
        f"onehot buckets={verif['val']['onehot_por_bucket']}")
    (VERIF / "preparar_ce.json").write_text(json.dumps(verif, indent=2))
    log("FIN preparar CE")

if __name__ == "__main__":
    main()
