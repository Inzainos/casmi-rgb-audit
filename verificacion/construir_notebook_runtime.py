#!/usr/bin/env python3
"""Construye casmi_runtime_notebook.ipynb: Baseline 1 que GENERA las
predicciones en runtime desde el test.parquet que reciba (público u oculto).

El código del pipeline es el de 01_construir.py (modo full) + 03_predecir_test.py
+ casmi_lib.py, adaptado: rutas parametrizadas (input de Kaggle con fallback
local), construcción de la librería en memoria (sin round-trip a npz), y salida
para TODOS los molecule_id del test recibido (en 03 los de espectros
descartados no aparecían; aquí llevan fallback "CCO" x25).
"""
import nbformat as nbf

NB_PATH = "/home/hatch/workspace/snt-taxonomia/kaggle-casmi/casmi_runtime_notebook.ipynb"

MD1 = """# CASMI 2026 — Baseline 1 en runtime (búsqueda en librería espectral)

**Método (idéntico al Baseline 1 verificado localmente, informe §2):** se construye la librería espectral bineada desde `train.parquet` (bins 0.01 Da, top-100 picos por espectro, precursor 100–2000 Da, ≥3 picos), y por cada molécula del `test.parquet` que el notebook reciba se buscan candidatos por precursor (±10 ppm) con similitud coseno (raíz cuadrada + L2, estándar GNPS), agregando por molécula con el máximo por InChIKey14 y devolviendo los 25 mejores SMILES representativos, en orden de similitud.

**Por qué en runtime:** en los concursos de código Kaggle re-ejecuta el notebook y sustituye `test.parquet` por el test oculto. Un archivo estático con los ids del test público no califica ("Submission Scoring Error" en los 2 envíos previos). Este notebook genera las predicciones desde el test que reciba, así que funciona con el test oculto.

**Cumplimiento:** sin internet; solo datos de la competencia (`train.parquet` + `test.parquet` del input); sin datos externos; sin RDKit en runtime (solo numpy/pandas/scipy/pyarrow, presentes en la imagen estándar de Kaggle). La salida es `submission.csv` con una fila por `molecule_id` del test recibido y exactamente 25 candidatos por fila."""

CODE1 = """import hashlib, resource, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

T0 = time.time()
def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time()-T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)

BIN_W = 0.01
NBINS = 200_000          # 0..2000 Da a 0.01 Da
TOPK = 100               # picos por espectro (por intensidad)
PMZ_MIN, PMZ_MAX = 100.0, 2000.0
NPEAK_MIN = 3
CHUNK = 100_000

def resolver_datos():
    # Input del concurso en Kaggle: /kaggle/input/<slug>/ con train+test.
    # Se busca por glob (el slug exacto puede variar) y se cae a datos/ locales.
    kagg = Path('/kaggle/input')
    if kagg.is_dir():
        for d in sorted(kagg.iterdir()):
            if d.is_dir() and (d / 'train.parquet').exists() and (d / 'test.parquet').exists():
                return d
        for p in sorted(kagg.rglob('train.parquet')):
            if (p.parent / 'test.parquet').exists():
                return p.parent
    for d in [Path('datos'),
              Path('/home/hatch/workspace/snt-taxonomia/kaggle-casmi/datos')]:
        if (d / 'train.parquet').exists() and (d / 'test.parquet').exists():
            return d
    raise FileNotFoundError('no se encontró train.parquet + test.parquet')

DATOS = resolver_datos()
WORK = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('verificacion/runtime_check')
WORK.mkdir(parents=True, exist_ok=True)
log(f"datos: {DATOS} | trabajo: {WORK}")
log(f"train: {(DATOS / 'train.parquet').stat().st_size / 1e9:.2f} GB | "
    f"test: {(DATOS / 'test.parquet').stat().st_size / 1e6:.1f} MB")"""

CODE2 = """# ---- Construcción de la librería (01_construir.py, modo full, en memoria) ----
def binear_fila(mzs, ints):
    # Top-100 por intensidad, bins de 0.01 Da, suma por bin. Devuelve (bins, vals).
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    ok = (ints > 0) & np.isfinite(mzs) & np.isfinite(ints) & (mzs < 2000.0) & (mzs >= 0)
    if not ok.any():
        return None
    mzs, ints = mzs[ok], ints[ok]
    if len(ints) > TOPK:
        idx = np.argpartition(-ints, TOPK)[:TOPK]
        mzs, ints = mzs[idx], ints[idx]
    bins = np.floor(mzs / BIN_W + 1e-9).astype(np.int32)
    bins = np.clip(bins, 0, NBINS - 1)
    ub, inv = np.unique(bins, return_inverse=True)
    vals = np.bincount(inv, weights=ints, minlength=len(ub)).astype(np.float32)
    return ub, vals

# Moléculas (códigos deterministas: inchikey14 ordenados, igual que fase_split)
ik = pq.read_table(str(DATOS / 'train.parquet'), columns=['inchikey14']).column('inchikey14').to_pandas()
vc = ik.value_counts()
mols = np.sort(vc.index.to_numpy())
mol_index = {m: i for i, m in enumerate(mols)}
del ik, vc
log(f"moléculas train: {len(mols):,}")

# Pasada escalar: cotas de filas/nnz de la librería completa (para prealocar)
pf = pq.ParquetFile(str(DATOS / 'train.parquet'))
rows_upper = 0
nnz_upper = 0
for batch in pf.iter_batches(batch_size=250_000, columns=['precursor_mz', 'num_peaks']):
    df = batch.to_pandas()
    pmz = df['precursor_mz'].to_numpy(dtype=np.float64)
    npeak = df['num_peaks'].to_numpy(dtype=np.int64)
    keep = (pmz >= PMZ_MIN) & (pmz <= PMZ_MAX) & (npeak >= NPEAK_MIN)
    rows_upper += int(keep.sum())
    nnz_upper += int(np.minimum(npeak, TOPK)[keep].sum())
    del df
log(f"pasada escalar: filas librería {rows_upper:,} | nnz cota {nnz_upper:,}")

# Construir CSR en memoria (misma lógica que construir(modo='full'))
R = np.empty(nnz_upper, dtype=np.int32)
C = np.empty(nnz_upper, dtype=np.int32)
D = np.empty(nnz_upper, dtype=np.float32)
prec = np.empty(rows_upper, dtype=np.float32)
mol = np.empty(rows_upper, dtype=np.int32)
off = 0
rowidx = 0
smiles_count = {}
stats = {'filas_leidas': 0, 'filtradas_pmz': 0, 'filtradas_npeak': 0,
         'sin_picos': 0, 'a_libreria': 0}
pf = pq.ParquetFile(str(DATOS / 'train.parquet'))
n_chunks = 0
for batch in pf.iter_batches(batch_size=CHUNK,
                             columns=['inchikey14', 'normalized_smiles', 'precursor_mz',
                                      'ms2_mzs', 'ms2_normalized_intensities', 'num_peaks']):
    df = batch.to_pandas()
    n_chunks += 1
    for pos in range(len(df)):
        ikv = df['inchikey14'].iloc[pos]
        stats['filas_leidas'] += 1
        pmz = float(df['precursor_mz'].iloc[pos])
        npeak = int(df['num_peaks'].iloc[pos])
        if not (PMZ_MIN <= pmz <= PMZ_MAX):
            stats['filtradas_pmz'] += 1
            continue
        if npeak < NPEAK_MIN:
            stats['filtradas_npeak'] += 1
            continue
        sm = df['normalized_smiles'].iloc[pos]
        d = smiles_count.setdefault(ikv, {})
        d[sm] = d.get(sm, 0) + 1
        r = binear_fila(df['ms2_mzs'].iloc[pos], df['ms2_normalized_intensities'].iloc[pos])
        if r is None:
            stats['sin_picos'] += 1
            continue
        bins, vals = r
        n = len(bins)
        R[off:off + n] = rowidx
        C[off:off + n] = bins
        D[off:off + n] = vals
        off += n
        prec[rowidx] = pmz
        mol[rowidx] = mol_index[ikv]
        rowidx += 1
        stats['a_libreria'] += 1
    del df
    log(f"chunk {n_chunks}: leídas {stats['filas_leidas']:,} | lib {stats['a_libreria']:,}")
X = sparse.csr_matrix((D[:off], (R[:off], C[:off])), shape=(rowidx, NBINS))
X.sum_duplicates()
prec_lib = prec[:rowidx]
mol_lib = mol[:rowidx]
rep = {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in smiles_count.items()}
del R, C, D, prec, mol, smiles_count
log(f"librería construida: {X.shape[0]:,} filas, {X.nnz:,} nnz | "
    f"rep SMILES: {len(rep):,} moléculas | {stats}")"""

CODE3 = """# ---- Funciones de similitud (casmi_lib.py, verbatim) ----
LN2 = np.log(2.0)
LN4 = np.log(4.0)

def g(x):
    out = np.zeros_like(x, dtype=np.float64)
    m = x > 0
    out[m] = x[m] * np.log(x[m])
    return out

def row_stats(X):
    rowsum = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    rowsum[rowsum <= 0] = 1.0
    nrows = X.shape[0]
    S = np.zeros(nrows, dtype=np.float64)
    indptr = X.indptr
    data = X.data.astype(np.float64)
    B = 200_000
    for s in range(0, nrows, B):
        e = min(nrows, s + B)
        sl = slice(indptr[s], indptr[e])
        d = data[sl]
        rep_r = np.repeat(rowsum[s:e], np.diff(indptr[s:e + 1]))
        p = d / rep_r
        S[s:e] = -np.bincount(np.repeat(np.arange(e - s), np.diff(indptr[s:e + 1])),
                              weights=np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0),
                              minlength=e - s)
    norm_cos = np.sqrt(rowsum)
    return rowsum, S, norm_cos

def candidate_range(prec_sorted, pmz):
    # Ventana de precursor: max(10 ppm, 0.002 Da).
    tol = max(pmz * 10e-6, 0.002)
    lo = int(np.searchsorted(prec_sorted, pmz - tol))
    hi = int(np.searchsorted(prec_sorted, pmz + tol))
    return lo, hi

def score_query(X, prec_sorted, mol_sorted, rowsum, S_row, norm_cos, q_row, q_pmz):
    lo, hi = candidate_range(prec_sorted, q_pmz)
    if hi <= lo:
        return None, None, None
    C = X[lo:hi]
    q_data = q_row.data.astype(np.float64)
    q_ind = q_row.indices
    q_sum = q_data.sum()
    if q_sum <= 0:
        return None, None, None
    q_sqrt = np.sqrt(q_data)
    q_norm_cos = np.sqrt(q_sum)
    q_l1 = q_data / q_sum
    S_q = -float(np.sum(np.where(q_l1 > 0, q_l1 * np.log(np.where(q_l1 > 0, q_l1, 1.0)), 0.0)))
    q_dense = np.zeros(X.shape[1], dtype=np.float64)
    q_dense[q_ind] = q_sqrt
    Csqrt = C.copy()
    Csqrt.data = np.sqrt(C.data.astype(np.float64)).astype(np.float32)
    cos = np.asarray(Csqrt.dot(q_dense)).ravel()
    cos = cos / (norm_cos[lo:hi] * q_norm_cos)
    np.clip(cos, -1.0, 1.0, out=cos)
    Cq = C[:, q_ind].toarray().astype(np.float64)
    rs = rowsum[lo:hi][:, None]
    pc = Cq / rs
    s = q_l1[None, :] + pc
    term = np.sum(g(s) - g(q_l1)[None, :] - g(pc), axis=1)
    S_AB = LN2 + (S_q + S_row[lo:hi]) / 2.0 - term / 2.0
    ent = 1.0 - (2.0 * S_AB - S_q - S_row[lo:hi]) / LN4
    np.clip(ent, 0.0, 1.0, out=ent)
    return mol_sorted[lo:hi], cos, ent

def rank_molecule(scores_por_espectro, mol_cand_por_espectro):
    best = {}
    for mol_cand, sc in zip(mol_cand_por_espectro, scores_por_espectro):
        for m, s in zip(mol_cand, sc):
            if s > best.get(m, -1.0):
                best[m] = float(s)
    return sorted(best, key=lambda m: -best[m])

# Ordenar la librería por precursor (igual que load_library) y calcular stats
order = np.argsort(prec_lib, kind='stable')
X = X[order]
prec_s = prec_lib[order].astype(np.float64)
mol_s = mol_lib[order]
del order, prec_lib, mol_lib
rowsum, S_row, norm_cos = row_stats(X)
log(f"librería ordenada por precursor + stats listas: {X.shape[0]:,} filas")"""

CODE4 = """# ---- Predicción sobre el test recibido (03_predecir_test.py) ----
test = pd.read_parquet(DATOS / 'test.parquet')
log(f"test recibido: {len(test):,} espectros, {test['molecule_id'].nunique():,} moléculas")

# Binear espectros de test (misma función que la librería)
rows, cols, data = [], [], []
q_mol_id, q_pmz = [], []
n_drop = 0
for i, r in test.iterrows():
    pmz = float(r['precursor_mz'])
    b = binear_fila(r['ms2_mzs'], r['ms2_normalized_intensities'])
    if b is None or not (100.0 <= pmz <= 2000.0):
        n_drop += 1
        continue
    bins, vals = b
    k = len(q_mol_id)
    rows.extend([k] * len(bins))
    cols.extend(bins.tolist())
    data.extend(vals.tolist())
    q_mol_id.append(r['molecule_id'])
    q_pmz.append(pmz)
Q = sparse.csr_matrix((np.asarray(data, dtype=np.float32),
                       (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
                      shape=(len(q_mol_id), 200_000))
Q.sum_duplicates()
q_mol_id = np.asarray(q_mol_id, dtype=object)
q_pmz = np.asarray(q_pmz, dtype=np.float64)
log(f"test bineado: {Q.shape[0]:,} espectros, {Q.nnz:,} nnz, descartados {n_drop}")

# Por molecule_id: agregar sus espectros (máxima similitud por inchikey14),
# top-25 y relleno a exactamente 25 repitiendo el top-1. TODOS los molecule_id
# del test recibido llevan fila (los sin espectros usables llevan "CCO" x25).
order = np.argsort(q_mol_id, kind='stable')
q_mol_s = q_mol_id[order]
uniq, starts = np.unique(q_mol_s, return_index=True)
bounds = list(starts) + [len(q_mol_s)]
por_mol = {}
t0 = time.time()
for ui, mid in enumerate(uniq):
    s, e = bounds[ui], bounds[ui + 1]
    best, mols_c = [], []
    for j in order[s:e]:
        m, c, _en = score_query(X, prec_s, mol_s, rowsum, S_row, norm_cos,
                                Q[j], float(q_pmz[j]))
        if m is None:
            continue
        best.append(c)
        mols_c.append(m)
    if not best:
        por_mol[mid] = ['CCO'] * 25
        continue
    top = rank_molecule(best, mols_c)
    smiles = []
    for code in top:
        ik = mols[code]
        sm = rep.get(ik)
        if sm:
            smiles.append(sm)
    if not smiles:
        smiles = ['CCO']
    smiles = smiles[:25]
    while len(smiles) < 25:
        smiles.append(smiles[0])
    por_mol[mid] = smiles
    if (ui + 1) % 100 == 0:
        log(f"predicción: {ui + 1:,}/{len(uniq):,} moléculas ({time.time() - t0:.0f}s)")

# Filas para TODOS los molecule_id del test (orden alfabético, determinista)
todos = sorted(test['molecule_id'].unique().tolist())
out_rows = [{'molecule_id': mid, 'smiles': ';'.join(por_mol.get(mid, ['CCO'] * 25))}
            for mid in todos]
sub = pd.DataFrame(out_rows)
sub.to_csv(WORK / 'submission.csv', index=False)
n_sm = sub['smiles'].apply(lambda s: len(s.split(';')))
log(f"submission escrita: {len(sub)} filas, SMILES por molécula: media {n_sm.mean():.1f}, "
    f"mediana {n_sm.median():.0f}, min {n_sm.min()}, max {n_sm.max()}, con 25: {(n_sm == 25).sum()}")"""

CODE5 = """# ---- Verificación de la salida (genérica: vale para el test público y el oculto) ----
sub2 = pd.read_csv(WORK / 'submission.csv')
test_ids = set(test['molecule_id'].unique().tolist())
n_c = sub2['smiles'].apply(lambda s: len(str(s).split(';')))
checks = {
    'columnas_exactas': list(sub2.columns) == ['molecule_id', 'smiles'],
    'filas_igual_a_moleculas_test': len(sub2) == len(test_ids),
    'ids_unicos': bool(sub2['molecule_id'].is_unique),
    'ids_iguales_a_test': set(sub2['molecule_id'].tolist()) == test_ids,
    'sin_nulos': bool(sub2['smiles'].notna().all() and sub2['molecule_id'].notna().all()),
    'sin_vacios': bool((sub2['smiles'].astype(str).str.strip() != '').all()),
    'exactamente_25_por_fila': bool((n_c == 25).all()),
}
md5 = hashlib.md5(open(WORK / 'submission.csv', 'rb').read()).hexdigest()
for k, v in checks.items():
    print(('OK   ' if v else 'FAIL ') + f'- {k}')
print(f"filas: {len(sub2)} | candidatos totales: {int(n_c.sum())} | "
      f"min/fila {int(n_c.min())} max/fila {int(n_c.max())}")
print(f"md5 submission.csv: {md5}")
print(f"tiempo total: {time.time() - T0:.1f}s")
assert all(checks.values()), 'verificación final con fallos'
print('VERIFICACIÓN FINAL: TODO OK')"""

MD2 = """## Notas de cumplimiento y de ejecución

- **Sin internet:** el notebook no usa red en ningún punto; la configuración de internet del notebook debe quedar desactivada al enviar (requisito del concurso).
- **Solo datos de la competencia:** `train.parquet` y `test.parquet` se leen del input del concurso (`/kaggle/input/...`); el fallback a rutas locales solo existe para la verificación en la máquina de desarrollo y nunca se activa en Kaggle.
- **Sin datos externos ni modelos preentrenados.**
- **Sin RDKit en runtime:** la búsqueda solo usa numpy/pandas/scipy/pyarrow.
- **Tiempo/memoria medidos en local:** build ~352 s + predicción ~20 s, RSS máx ~2.9 GB (límite del notebook: 9 h; CPU Kaggle ~13 GB RAM).
- **Relleno a 25:** cada fila trae exactamente 25 candidatos; si una molécula tiene menos de 25 candidatos reales en su ventana, se repite su mejor candidato (el sample del concurso trae 25 "CCO" idénticos por fila, así que los duplicados son aceptados por el formato). Si una molécula no tiene ningún candidato, se usa "CCO".
- **Salida:** `submission.csv` en el working dir, una fila por cada `molecule_id` del test recibido, en orden alfabético de `molecule_id`."""

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(MD1),
    nbf.v4.new_code_cell(CODE1),
    nbf.v4.new_code_cell(CODE2),
    nbf.v4.new_code_cell(CODE3),
    nbf.v4.new_code_cell(CODE4),
    nbf.v4.new_code_cell(CODE5),
    nbf.v4.new_markdown_cell(MD2),
]
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.12.3"}
nbf.write(nb, NB_PATH)
print(f"notebook escrito: {NB_PATH} ({len(nb.cells)} celdas, "
      f"{sum(1 for c in nb.cells if c.cell_type == 'code')} de código)")
