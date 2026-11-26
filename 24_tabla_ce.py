"""CASMI — 24_tabla_ce.py
Tabla de energía de colisión (CE) por espectro, alineada fila a fila con las
CSR de validación y con la librería completa.

Por qué existe: el candidato (b) del cap. 37 (wpc-schumann-ideas-cap37.md)
— "energía de colisión como variable" — necesita la CE de cada espectro, y
`collision_energy_ev` NO existe en ningún archivo de `pruebas/` (ronda 2,
saltado por eso). Esta tabla se construye desde train.parquet en esta caja
y se sube a `pruebas/` para el experimento CE-como-feature.

Qué guarda (un .npz por destino, mismas filas y mismo orden que la CSR):
  ce_n     uint8    nº de valores CE en la lista (0 = sin CE)
  ce_first float32  primer valor de la lista (NaN si n=0)
  ce_mean  float32  media de la lista (NaN si n=0)
  ce_min   float32  mínimo de la lista (NaN si n=0)
  ce_max   float32  máximo de la lista (NaN si n=0)
  precursor_mz float32  (copia, para verificar la alineación contra la CSR)
  mol      int32    código de molécula (copia, para verificar la alineación)

La CE es una LISTA por espectro (no un escalar): 67% tiene 1 valor, 15.5%
tiene 3 (típico '20,40,60'), 13.3% no tiene ninguno. Por eso la tabla
guarda n/first/mean/min/max en vez de imponer una reducción: el experimento
elige la reducción (primera, media, o modelo por CE) sin releer el parquet.

Alineación (reproduce EXACTO el destino de 01_construir.py, modo val/full):
  - orden de filas = orden de train.parquet, filtrado por destino;
  - filtro: precursor_mz en [100, 2000] y num_peaks >= 3;
  - split semilla 42 desde verificacion/split_arrays.npz (val 5% moléculas,
    spec 5% de las de >=4 espectros, disjuntos);
  - val: qmol si h < 3/max(n_mol,1) o n_mol <= 3 (el resto de esa molécula
    se DESCARTA, no va a librería); spec: qspec si holdout_spec[g],
    si no lib; resto lib. h = ((g*2654435761) % 2^32)/2^32, g = fila global;
  - filtro de picos de binear_fila: la fila se descarta si NINGÚN pico
    pasa (ints>0, m/z e intensidad finitos, 0<=m/z<2000). Vectorizado con
    offsets de pyarrow (no se reimplementa el top-100: solo importa si
    hay al menos un pico válido).
  - full: lib para toda fila que pasa los filtros.

Verificación (en este mismo script, antes de escribir nada):
  - conteo de filas por destino == shape de la CSR correspondiente;
  - precursor_mz y mol, elemento a elemento, == meta de la CSR.
  Si algo no coincide, NO escribe los archivos y falla con el detalle.

Salida: intermedios/ce_{val_lib,val_qmol,val_qspec,full_lib}.npz
        verificacion/tabla_ce.json (conteos, distribución CE, verificación)
        verificacion/tabla_ce.log  (este log)
Uso: ~/workspace/py313-venv/bin/python 24_tabla_ce.py
"""
import gc
import json, time, resource
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

T0 = time.time()

def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time()-T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)

BASE = Path("/home/hatch/workspace/snt-taxonomia/kaggle-casmi")
REPO = Path("/home/hatch/workspace/enveda-casmi-2026")
DATOS = BASE / "datos"
INTER = REPO / "intermedios"
VERIF = BASE / "verificacion"
LOGFILE = REPO / "verificacion" / "tabla_ce.log"

PMZ_MIN, PMZ_MAX = 100.0, 2000.0
NPEAK_MIN = 3
CHUNK = 50_000

# --- split (caché de 01_construir.py, semilla 42) ---
z = np.load(VERIF / "split_arrays.npz", allow_pickle=True)
mols = z["mols"]
n_per_mol = dict(zip(mols.tolist(), z["n_per_mol"].tolist()))
val_set = set(z["val_mols"].tolist())
spec_set = set(z["spec_mols"].tolist())
n_per_mol_arr = np.asarray([n_per_mol[m] for m in mols], dtype=np.int64)
is_val_code = np.zeros(len(mols), dtype=bool)
is_spec_code = np.zeros(len(mols), dtype=bool)
for m in val_set:
    is_val_code[np.searchsorted(mols, m)] = True
for m in spec_set:
    is_spec_code[np.searchsorted(mols, m)] = True
holdout = np.load(VERIF / "holdout_spec.npy")
log(f"split: {len(mols):,} moléculas, val {len(val_set):,}, spec {len(spec_set):,}, "
    f"holdout {int(holdout.sum()):,} filas")

# --- pasada principal: destino por fila + CE ---
dests = ["lib", "qmol", "qspec", "full"]
acc = {d: {"ce_n": [], "ce_first": [], "ce_mean": [], "ce_min": [], "ce_max": [],
           "pmz": [], "mol": []} for d in dests}
stats = {"filas_leidas": 0, "filtradas_pmz": 0, "filtradas_npeak": 0, "sin_picos": 0,
         "a_lib": 0, "a_qmol": 0, "a_qspec": 0, "a_full": 0}
ce_min_global, ce_max_global, ce_neg = np.inf, -np.inf, 0

pf = pq.ParquetFile(str(DATOS / "train.parquet"))
offset = 0
for batch in pf.iter_batches(batch_size=CHUNK, columns=[
        "inchikey14", "precursor_mz", "num_peaks",
        "ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"]):
    n = batch.num_rows
    # códigos de molécula sin pasar por pandas: dictionary-encode + searchsorted
    ik_enc = pc.dictionary_encode(batch.column("inchikey14"))
    dict_vals = np.asarray(ik_enc.dictionary.to_pylist())
    dict_codes = np.searchsorted(mols, dict_vals)
    codes = dict_codes[np.asarray(ik_enc.indices.to_pylist(), dtype=np.int64)].astype(np.int32)
    pmz = batch.column("precursor_mz").to_numpy().astype(np.float64)
    npeak = batch.column("num_peaks").to_numpy().astype(np.int64)
    keep = (pmz >= PMZ_MIN) & (pmz <= PMZ_MAX) & (npeak >= NPEAK_MIN) & (codes >= 0)
    stats["filtradas_pmz"] += int(((pmz < PMZ_MIN) | (pmz > PMZ_MAX)).sum())
    stats["filtradas_npeak"] += int(((pmz >= PMZ_MIN) & (pmz <= PMZ_MAX) & (npeak < NPEAK_MIN)).sum())
    c = np.clip(codes, 0, None)
    is_val = is_val_code[c]
    is_spec = is_spec_code[c]
    n_mol = n_per_mol_arr[c]
    g = np.arange(offset, offset + n, dtype=np.int64)
    h = ((g * 2654435761) % (2 ** 32)) / (2 ** 32)
    thr = 3.0 / np.maximum(n_mol, 1)
    m_qmol = keep & is_val & ((h < thr) | (n_mol <= 3))
    m_qspec = keep & is_spec & holdout[offset:offset + n]
    m_lib = keep & (~is_val) & (~m_qspec)
    m_full = keep
    # filtro de picos de binear_fila (vectorizado): ¿algún pico válido?
    mz_arr = batch.column("ms2_mzs")
    it_arr = batch.column("ms2_normalized_intensities")
    flat_mz = mz_arr.values.to_numpy().astype(np.float64)
    flat_it = it_arr.values.to_numpy().astype(np.float64)
    valid_flat = (flat_it > 0) & np.isfinite(flat_mz) & np.isfinite(flat_it) \
        & (flat_mz >= 0) & (flat_mz < 2000.0)
    lens = mz_arr.value_lengths().to_numpy(zero_copy_only=False)
    row_idx = np.repeat(np.arange(n), lens)
    has_peak = np.bincount(row_idx, weights=valid_flat.astype(np.float64),
                           minlength=n) > 0
    stats["sin_picos"] += int((keep & ~has_peak).sum())
    m_qmol &= has_peak
    m_qspec &= has_peak
    m_lib &= has_peak
    m_full &= has_peak
    # CE por fila (listas), todo vectorizado sobre los valores aplanados
    ce_arr = batch.column("collision_energy_ev")
    ce_flat = ce_arr.values.to_numpy().astype(np.float64)
    ce_lens = ce_arr.value_lengths().to_numpy(zero_copy_only=False)
    ce_offs = ce_arr.offsets.to_numpy(zero_copy_only=False)
    ce_null = ce_arr.is_null().to_numpy(zero_copy_only=False)
    ce_n = np.where(ce_null, 0, ce_lens).astype(np.int64)
    has_ce = ce_n > 0
    ce_first = np.full(n, np.nan, dtype=np.float64)
    ce_mean = np.full(n, np.nan, dtype=np.float64)
    ce_min = np.full(n, np.nan, dtype=np.float64)
    ce_max = np.full(n, np.nan, dtype=np.float64)
    if ce_flat.size:
        # reduceat con TODOS los offsets: para filas vacías devuelve un valor
        # basura que se enmascara a NaN después; para filas con CE el slice
        # [offs[i], offs[i+1]) es exactamente su lista. Los índices se
        # recortan a len(flat)-1 porque reduceat no acepta offs == len(flat)
        # (filas vacías al final del chunk); el recorte solo toca filas
        # vacías, que se enmascaran.
        red_idx = np.minimum(ce_offs[:-1], len(ce_flat) - 1)
        ce_first[has_ce] = ce_flat[ce_offs[:-1][has_ce]]
        ce_mean[has_ce] = np.add.reduceat(ce_flat, red_idx)[has_ce] / ce_n[has_ce]
        ce_min[has_ce] = np.minimum.reduceat(ce_flat, red_idx)[has_ce]
        ce_max[has_ce] = np.maximum.reduceat(ce_flat, red_idx)[has_ce]
        if ce_flat.min() < ce_min_global:
            ce_min_global = float(ce_flat.min())
        if ce_flat.max() > ce_max_global:
            ce_max_global = float(ce_flat.max())
        ce_neg += int((ce_flat < 0).sum())
    for m, name in ((m_lib, "lib"), (m_qmol, "qmol"), (m_qspec, "qspec"), (m_full, "full")):
        a = acc[name]
        a["ce_n"].append(ce_n[m].astype(np.uint8))
        a["ce_first"].append(ce_first[m].astype(np.float32))
        a["ce_mean"].append(ce_mean[m].astype(np.float32))
        a["ce_min"].append(ce_min[m].astype(np.float32))
        a["ce_max"].append(ce_max[m].astype(np.float32))
        a["pmz"].append(pmz[m].astype(np.float32))
        a["mol"].append(codes[m].astype(np.int32))
        stats["a_" + name] += int(m.sum())
    stats["filas_leidas"] += n
    offset += n
    del batch
    gc.collect()  # la caja tiene ~6 GB; soltar los buffers del chunk
    log(f"chunk: leídas {stats['filas_leidas']:,} | lib {stats['a_lib']:,} | "
        f"qmol {stats['a_qmol']:,} | qspec {stats['a_qspec']:,} | full {stats['a_full']:,}")

log(f"FIN pasada: {stats}")
log(f"CE global: min {ce_min_global}, max {ce_max_global}, valores negativos {ce_neg}")

# --- verificación contra las CSR (conteos + pmz/mol elemento a elemento) ---
csr_map = {"lib": "val_lib", "qmol": "val_qmol", "qspec": "val_qspec", "full": "full_lib"}
verif_ok = True
verif = {}
for name, tag in csr_map.items():
    a = acc[name]
    ce_n = np.concatenate(a["ce_n"]) if a["ce_n"] else np.zeros(0, dtype=np.uint8)
    pmz_a = np.concatenate(a["pmz"]) if a["pmz"] else np.zeros(0, dtype=np.float32)
    mol_a = np.concatenate(a["mol"]) if a["mol"] else np.zeros(0, dtype=np.int32)
    csr = sparse.load_npz(INTER / f"{tag}.npz")
    meta = np.load(INTER / f"{tag}_meta.npz", allow_pickle=True)
    n_csr = csr.shape[0]
    ok_count = len(ce_n) == n_csr
    ok_pmz = ok_count and np.array_equal(pmz_a, meta["precursor_mz"])
    ok_mol = ok_count and np.array_equal(mol_a, meta["mol"])
    verif[name] = {"filas_tabla": int(len(ce_n)), "filas_csr": int(n_csr),
                   "conteo_ok": bool(ok_count), "pmz_exacto": bool(ok_pmz),
                   "mol_exacto": bool(ok_mol)}
    log(f"verif {tag}: tabla {len(ce_n):,} vs CSR {n_csr:,} | conteo {ok_count} | "
        f"pmz {ok_pmz} | mol {ok_mol}")
    if not (ok_count and ok_pmz and ok_mol):
        verif_ok = False
        if ok_count and not ok_pmz:
            d = np.where(pmz_a != meta["precursor_mz"])[0]
            log(f"  pmz difiere en {len(d)} filas, primera {d[0] if len(d) else None}")
        if ok_count and not ok_mol:
            d = np.where(mol_a != meta["mol"])[0]
            log(f"  mol difiere en {len(d)} filas, primera {d[0] if len(d) else None}")

if not verif_ok:
    log("VERIFICACIÓN FALLIDA — no se escribe nada")
    raise SystemExit(1)

# --- escritura ---
salidas = {}
for name, tag in csr_map.items():
    a = acc[name]
    out = {
        "ce_n": np.concatenate(a["ce_n"]),
        "ce_first": np.concatenate(a["ce_first"]),
        "ce_mean": np.concatenate(a["ce_mean"]),
        "ce_min": np.concatenate(a["ce_min"]),
        "ce_max": np.concatenate(a["ce_max"]),
        "precursor_mz": np.concatenate(a["pmz"]),
        "mol": np.concatenate(a["mol"]),
    }
    p = INTER / f"ce_{tag}.npz"
    np.savez(p, **out)
    n = len(out["ce_n"])
    ce_n = out["ce_n"]
    salidas[tag] = {
        "filas": int(n),
        "sin_ce_frac": float((ce_n == 0).mean()),
        "un_valor_frac": float((ce_n == 1).mean()),
        "multi_frac": float((ce_n >= 2).mean()),
        "ce_first_mediana": float(np.nanmedian(out["ce_first"])),
        "ce_mean_mediana": float(np.nanmedian(out["ce_mean"])),
        "archivo_bytes": int(p.stat().st_size),
    }
    log(f"escrito {p.name}: {n:,} filas, {p.stat().st_size/1e6:.1f} MB, "
        f"sin CE {salidas[tag]['sin_ce_frac']:.3f}, 1 valor {salidas[tag]['un_valor_frac']:.3f}, "
        f"multi {salidas[tag]['multi_frac']:.3f}")

informe = {"stats": stats, "verificacion": verif,
           "ce_global": {"min": ce_min_global, "max": ce_max_global,
                         "valores_negativos": int(ce_neg)},
           "salidas": salidas,
           "nota": ("CE es lista por espectro; la tabla guarda n/first/mean/min/max "
                    "para que el experimento elija la reducción sin releer train.parquet. "
                    "Alineación verificada elemento a elemento (pmz+mol) contra las CSR.")}
(VERIF / "tabla_ce.json").write_text(json.dumps(informe, indent=2))
log("informe en verificacion/tabla_ce.json — OK")
