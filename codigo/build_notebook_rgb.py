"""Genera casmi_runtime_notebook_rgb.ipynb desde sus celdas (fuente reproducible).
El notebook resultante es autocontenido (sin internet) y sigue la estructura
del notebook de la submission 3 (casmi_runtime_notebook.ipynb): resuelve el
input de la competencia y el dataset privado adjunto, binea el test recibido,
predice huellas con el lineal RGB, aplica lineal_bin, recupera por Tanimoto
con filtro de masa y escribe submission.csv.

Verificación local: ejecutar con nbconvert (venv casmi-nb-venv, kernel casmi-nb)
desde ~/workspace/enveda-casmi-2026; la salida debe ser byte-idéntica a
salida/submission_rgb.csv (md5 6dcceab5d57a00b82d485439ebd6061b).
Uso: python3 build_notebook_rgb.py
"""
import json
from pathlib import Path

BASE = Path("/home/hatch/workspace/enveda-casmi-2026")

MD_MD5_LOCAL = "6dcceab5d57a00b82d485439ebd6061b"

cells_src = []

cells_src.append(("markdown", """# CASMI 2026 — RGB 3 canales en runtime (espectro → huella Morgan, variante bin)

**Método (idéntico al modelo validado localmente, MRR@25 mol-split 0.4951):** el espectro de cada molécula del `test.parquet` que el notebook reciba se binea (0.01 Da, top-100 picos por espectro) y se transforma en 3 canales por bin m/z — intensidad con raíz cuadrada, log1p(intensidad) y presencia binaria — cada canal normalizado por fila. La matriz resultante (3 × 20,000 columnas) se multiplica por la matriz lineal `W_lineal_rgb20k.npy` (60,000 × 2,048) para predecir la huella de Morgan de 2,048 bits; las predicciones de los espectros de una molécula se promedian. Se aplica la variante lineal_bin (clip [0,1] + top-100 bits a 1) y se recuperan los 25 mejores SMILES del pool de train por similitud de Tanimoto, con filtro de masa de precursor (±10 ppm, piso 0.002 Da).

**Por qué en runtime:** en los concursos de código Kaggle re-ejecuta el notebook y sustituye `test.parquet` por el test oculto. Un archivo estático con los ids del test público no califica ("Submission Scoring Error" en los 2 primeros envíos). Este notebook genera las predicciones desde el test que reciba, así que funciona con el test oculto.

**Cumplimiento:** sin internet; solo datos de la competencia (`test.parquet` del input) + un dataset privado adjunto con el modelo y el pool (W, bins, huellas, índice de masas, mapa de SMILES); sin datos externos; sin RDKit en runtime (solo numpy/pandas/scipy/pyarrow, presentes en la imagen estándar de Kaggle). La salida es `submission.csv` con una fila por `molecule_id` del test recibido y exactamente 25 candidatos por fila.
"""))

cells_src.append(("code", """import hashlib, resource, time
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
TOPK_REC = 40            # candidatos a recuperar antes de deduplicar SMILES
TOPK_OUT = 25

def resolver_competicion():
    # Input del concurso en Kaggle: /kaggle/input/<slug>/ con test.parquet.
    kagg = Path('/kaggle/input')
    if kagg.is_dir():
        for d in sorted(kagg.iterdir()):
            if d.is_dir() and (d / 'test.parquet').exists():
                return d
        for p in sorted(kagg.rglob('test.parquet')):
            return p.parent
    for d in [Path('datos'),
              Path('/home/hatch/workspace/snt-taxonomia/kaggle-casmi/datos')]:
        if (d / 'test.parquet').exists():
            return d
    raise FileNotFoundError('no se encontró test.parquet')

def resolver_dataset():
    # Dataset privado adjunto: contiene W_lineal_rgb20k.npy (+ bins, pool...).
    kagg = Path('/kaggle/input')
    if kagg.is_dir():
        for d in sorted(kagg.iterdir()):
            if d.is_dir() and (d / 'W_lineal_rgb20k.npy').exists():
                return d
        for p in sorted(kagg.rglob('W_lineal_rgb20k.npy')):
            return p.parent
    for d in [Path('runtime_rgb_dataset'), Path('intermedios')]:
        if (d / 'W_lineal_rgb20k.npy').exists():
            return d
    raise FileNotFoundError('no se encontró W_lineal_rgb20k.npy (dataset RGB)')

COMP = resolver_competicion()
DS = resolver_dataset()
WORK = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('verificacion/runtime_check_rgb')
WORK.mkdir(parents=True, exist_ok=True)
log(f"competición: {COMP} | dataset RGB: {DS} | trabajo: {WORK}")
log(f"test: {(COMP / 'test.parquet').stat().st_size / 1e6:.1f} MB | "
    f"W: {(DS / 'W_lineal_rgb20k.npy').stat().st_size / 1e6:.0f} MB")
"""))

cells_src.append(("code", """# ---- Bineado del test recibido (01_construir.binear_fila, verbatim) ----
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

test = pd.read_parquet(COMP / 'test.parquet')
log(f"test recibido: {len(test):,} espectros, {test['molecule_id'].nunique():,} moléculas")

rows, cols, data = [], [], []
q_mol_id, q_pmz = [], []
n_drop = 0
for i, r in test.iterrows():
    pmz = float(r['precursor_mz'])
    b = binear_fila(r['ms2_mzs'], r['ms2_normalized_intensities'])
    if b is None or not (PMZ_MIN <= pmz <= PMZ_MAX):
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
                      shape=(len(q_mol_id), NBINS))
Q.sum_duplicates()
q_mol_id = np.asarray(q_mol_id, dtype=object)
q_pmz = np.asarray(q_pmz, dtype=np.float64)
log(f"test bineado: {Q.shape[0]:,} espectros, {Q.nnz:,} nnz, descartados {n_drop}")
"""))

cells_src.append(("code", """# ---- Transformación RGB (21_entrenar_lineal_rgb.transformar_rgb, verbatim) ----
def l2_por_fila(X):
    \"\"\"Igual que en 14/21: dividir por sqrt(suma de valores de la fila).\"\"\"
    norms = np.sqrt(np.asarray(X.sum(axis=1)).ravel().astype(np.float32))
    norms[norms <= 0] = 1.0
    X = sparse.diags((1.0 / norms).astype(np.float32)) @ X
    return X.tocsr().astype(np.float32)

def transformar_rgb(Q, bins_sel):
    \"\"\"3 canales (sqrt / log1p / binario), cada uno normalizado por fila,
    concatenados [sqrt | log | bin]. Q es la librería bineada cruda.\"\"\"
    Qb = Q[:, bins_sel].tocsr()
    canales = []
    for modo in ("sqrt", "log", "bin"):
        Xc = Qb.copy().tocsr()
        if modo == "sqrt":
            Xc.data = np.sqrt(Xc.data.astype(np.float32)).astype(np.float32)
        elif modo == "log":
            Xc.data = np.log1p(Xc.data.astype(np.float32)).astype(np.float32)
        else:
            Xc.data = np.ones_like(Xc.data, dtype=np.float32)
        canales.append(l2_por_fila(Xc))
    del Qb
    return sparse.hstack(canales, format="csr").astype(np.float32)

W = np.load(DS / 'W_lineal_rgb20k.npy')
bins_sel = np.load(DS / 'bins_rgb.npy')
assert W.shape == (3 * len(bins_sel), 2048), (W.shape, len(bins_sel))
log(f"W RGB: {W.shape} ({W.nbytes/1e6:.0f} MB), bins_rgb {len(bins_sel):,}")
Qf = transformar_rgb(Q, bins_sel)
del Q
log(f"Qf RGB: {Qf.shape[0]:,} x {Qf.shape[1]:,}, nnz {Qf.nnz:,}")
P_spec = np.asarray(Qf @ W, dtype=np.float32)
del Qf, W
log(f"P test RGB por espectro: {P_spec.shape}, media {P_spec.mean():.4f} max {P_spec.max():.3f}")

# Agregar por molecule_id (media de sus espectros, igual que 17/09)
order = np.argsort(q_mol_id, kind="stable")
q_s = q_mol_id[order]
uniq, starts = np.unique(q_s, return_index=True)
bounds = list(starts) + [len(q_s)]
counts = np.diff(bounds).astype(np.float32)
rows_a = np.repeat(np.arange(len(uniq)), counts.astype(np.int64))
cols_a = order.astype(np.int32)
data_a = np.repeat((1.0 / counts).astype(np.float32), counts.astype(np.int64))
A = sparse.csr_matrix((data_a, (rows_a, cols_a)), shape=(len(uniq), len(q_mol_id)))
P_test = np.asarray(A @ P_spec, dtype=np.float32)
del P_spec, A
pos_en_P = {mid: i for i, mid in enumerate(uniq)}
log(f"P test por molécula: {P_test.shape}, espectros/mol media {counts.mean():.2f}")
"""))

cells_src.append(("code", """# ---- Variante lineal_bin (la validada: clip [0,1] + top-100 bits a 1) ----
P_bin = np.zeros_like(P_test)
for i in range(P_test.shape[0]):
    p = np.clip(P_test[i], 0.0, 1.0)
    P_bin[i, np.argpartition(-p, 100)[:100]] = 1.0
log("lineal_bin aplicado: clip [0,1] + top-100 bits a 1 (igual que 15 --bin y 17)")
del P_test

# ---- Retrieval con filtro de masa (13_evaluar_masa, verbatim) ----
def tanimoto_vec(p, F, nbits_c):
    \"\"\"Tanimoto continuo entre p (2048,) y las filas de F (C,2048) float32.\"\"\"
    dots = F @ p
    p2 = float(p @ p)
    den = p2 + nbits_c - dots
    t = np.zeros(len(dots), dtype=np.float32)
    m = den > 0
    t[m] = dots[m] / den[m]
    return t

def topk_de(t, codes, k=TOPK_OUT):
    \"\"\"Top-k índices de t (desc) y sus códigos.\"\"\"
    if len(t) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    kk = min(k, len(t))
    if kk < len(t):
        part = np.argpartition(-t, kk - 1)[:kk]
    else:
        part = np.arange(len(t))
    part = part[np.argsort(-t[part])]
    return codes[part].astype(np.int32), t[part].astype(np.float32)

pool = sparse.load_npz(DS / 'huellas_pool.npz').tocsr()
pool_nbits = np.diff(pool.indptr).astype(np.float32)
idx = np.load(DS / 'precursores_full_idx.npz')
prec_sorted, mol_sorted = idx["prec"], idx["mol"]
mapa = pd.read_parquet(DS / 'mapa_smiles_full.parquet')
mol_names = np.load(DS / 'mol_index.npy', allow_pickle=True)
rep = dict(zip(mapa["inchikey14"], mapa["smiles_rep"]))
log(f"pool {pool.shape[0]:,} moléculas, mapa SMILES {len(rep):,}")

# Precursores distintos por molécula de consulta (igual que 12_masas,
# pero desde los espectros bineados del test recibido). Se iteran
# ORDENADOS (como query_prec_test.npz): el orden de inserción en el
# conjunto de candidatos determina el orden de `codes` y, con los
# empates exactos de Tanimoto (huella binaria), el desempate de topk_de.
# Sin este orden, el notebook y 23_predecir_test_rgb.py discrepan en
# las filas con empates (verificado: 25/400 filas).
prec_por_mol = {}
for mid, pmz in zip(q_mol_id, q_pmz):
    prec_por_mol.setdefault(mid, set()).add(float(pmz))

# Por molecule_id: top-25 SMILES, relleno a exactamente 25 repitiendo el
# top-1. TODOS los molecule_id del test recibido llevan fila (los sin
# espectros usables o sin candidatos llevan "CCO" x25).
todos = sorted(test['molecule_id'].unique().tolist())
por_mol = {}
n_relleno_filas = 0
n_cco = 0
t0 = time.time()
for ui, mid in enumerate(todos):
    if mid not in pos_en_P:
        por_mol[mid] = ['CCO'] * TOPK_OUT
        n_cco += 1
        continue
    cands = set()
    for q in sorted(prec_por_mol[mid]):
        tol = max(float(q) * 10e-6, 0.002)
        lo = int(np.searchsorted(prec_sorted, q - tol))
        hi = int(np.searchsorted(prec_sorted, q + tol))
        if hi > lo:
            cands.update(mol_sorted[lo:hi].tolist())
    if not cands:
        por_mol[mid] = ['CCO'] * TOPK_OUT
        n_cco += 1
        continue
    codes = np.fromiter(cands, dtype=np.int32)
    F = pool[codes].toarray().astype(np.float32)
    nb = pool_nbits[codes]
    t = tanimoto_vec(P_bin[pos_en_P[mid]], F, nb)
    top_c, _ = topk_de(t, codes, k=TOPK_REC)
    smiles = []
    vistos = set()
    for code in top_c:
        ik = mol_names[code]
        sm = rep.get(ik)
        if not sm:
            continue
        if sm not in vistos:
            vistos.add(sm)
            smiles.append(sm)
        if len(smiles) >= TOPK_OUT:
            break
    if not smiles:
        por_mol[mid] = ['CCO'] * TOPK_OUT
        n_cco += 1
        continue
    if len(smiles) < TOPK_OUT:
        n_relleno_filas += 1
        smiles += [smiles[0]] * (TOPK_OUT - len(smiles))
    por_mol[mid] = smiles
    if (ui + 1) % 100 == 0:
        log(f"retrieval: {ui + 1:,}/{len(todos):,} moléculas ({time.time() - t0:.0f}s)")

out_rows = [{'molecule_id': mid, 'smiles': ';'.join(por_mol[mid])} for mid in todos]
sub = pd.DataFrame(out_rows)
sub.to_csv(WORK / 'submission.csv', index=False)
n_sm = sub['smiles'].apply(lambda s: len(s.split(';')))
log(f"submission escrita: {len(sub)} filas, SMILES por molécula: media {n_sm.mean():.1f}, "
    f"mediana {n_sm.median():.0f}, min {n_sm.min()}, max {n_sm.max()}, con 25: {(n_sm == 25).sum()}, "
    f"filas con relleno: {n_relleno_filas}, filas CCO: {n_cco}")
"""))

cells_src.append(("code", """# ---- Verificación de la salida (genérica: vale para el test público y el oculto) ----
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
print('VERIFICACIÓN FINAL: TODO OK')
"""))

cells_src.append(("markdown", f"""## Notas de cumplimiento y de ejecución

- **Sin internet:** el notebook no usa red en ningún punto; la configuración de internet del notebook debe quedar desactivada al enviar (requisito del concurso).
- **Solo datos de la competencia + dataset privado:** `test.parquet` se lee del input del concurso (`/kaggle/input/...`); el modelo y el pool se leen de un dataset privado adjunto (`W_lineal_rgb20k.npy`, `bins_rgb.npy`, `huellas_pool.npz`, `mol_index.npy`, `mapa_smiles_full.parquet`, `precursores_full_idx.npz`). Los fallbacks a rutas locales solo existen para la verificación en la máquina de desarrollo y nunca se activan en Kaggle.
- **Sin datos externos ni modelos preentrenados externos.** El modelo lineal se entrenó solo con el train de la competencia.
- **Sin RDKit en runtime:** solo numpy/pandas/scipy/pyarrow.
- **Verificación local:** ejecutado con el test visible; salida byte-idéntica a `salida/submission_rgb.csv` (md5 {MD_MD5_LOCAL}), 400 filas × 25 SMILES. Tiempo y memoria medidos en local: ver `verificacion/runtime_rgb.json` y el log de verificación.
- **Relleno a 25:** cada fila trae exactamente 25 candidatos; si una molécula tiene menos de 25 candidatos reales en su ventana, se repite su mejor candidato (el sample del concurso trae 25 "CCO" idénticos por fila, así que los duplicados son aceptados por el formato). Si una molécula no tiene ningún candidato, se usa "CCO". En el test visible, 33 filas necesitaron relleno (+359 repeticiones), 0 filas CCO.
- **Salida:** `submission.csv` en el working dir, una fila por cada `molecule_id` del test recibido, en orden alfabético de `molecule_id`.
- **Limitación conocida:** el orden entre candidatos con similitud de Tanimoto exactamente empatada depende del `argpartition` de numpy; con puntuaciones continuas float32 los empates exactos son raros y no afectan al MRR medido. OJO: con la variante bin la huella es binaria y los empates SON frecuentes (en el test visible, 25/400 filas cambian de orden si los precursores se iteran sin ordenar). Por eso el notebook itera los precursores de cada molécula ORDENADOS, igual que `query_prec_test.npz` en `23_predecir_test_rgb.py`, para que el desempate sea idéntico en ambos.
"""))

CELL_IDS = ["a1b2c3d4", "e5f6a7b8", "c9d0e1f2", "a3b4c5d6", "e7f8a9b0", "c1d2e3f4", "a5b6c7d8"]

def main():
    cells = []
    for (kind, src), cid in zip(cells_src, CELL_IDS):
        lines = src.splitlines(keepends=True)
        lines = [ln if ln.endswith("\n") else ln + "\n" for ln in lines]
        cell = {"cell_type": kind, "id": cid, "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = BASE / "casmi_runtime_notebook_rgb.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"notebook escrito: {out} ({out.stat().st_size} B, {len(cells)} celdas)")

if __name__ == "__main__":
    main()
