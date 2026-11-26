#!/usr/bin/env python3
"""Construye verificacion/casmi_notebook_generativo_v1.ipynb: notebook de
envío Kaggle del camino generativo CASMI (familia B, híbrido, protocolo
PROTOCOLO_GENERATIVO_DE_NOVO.md §§6/9, Fase 3/4, puertas G0-G5).

Diseño (decisiones documentadas, no bloqueantes):
- Híbrido retrieval-primero (protocolo §6): el retrieval ocupa las primeras
  posiciones solo cuando su confianza es alta (top-1 coseno >= 0.95, regla
  medida en el blend); por debajo, el generativo va primero y el retrieval
  rellena. La ablación generativo-primero queda como variante, no se
  implementa aquí.
- Etapa 1 = kNN espectro->huella (paso 2, Tanimoto 0.3148 honesto medido):
  top-10 moléculas por coseno, media ponderada de sus huellas del pool.
  Se eligió kNN sobre lineal/RGB porque reutiliza la librería que el
  notebook ya construye para el retrieval y su calidad (0.31) es justo el
  nivel de corrupción con el que se entrenó el decodificador.
- Binarización de la huella predicha: umbral 0.5 (medido en
  verificacion/medir_binarizacion_etapa1.py: Tanimoto 0.4855 vs 0.4617 de
  top-100, y 24 bits/fila, dentro de la distribución de entrenamiento).
- Decodificador: script fixeado (beam, v2) + checkpoint re-entrenado,
  cargados desde el dataset de Kaggle (/kaggle/input/...), nunca por
  internet. Beam-25 por molécula, lotes de 32.
- Dedup por InChIKey14 tautomer-canonicalizado (igual que el evaluador
  30_evaluar_generativo.py) cuando hay RDKit; si no, por SMILES exacto.
  RDKit se instala offline desde la rueda del dataset (mejor esfuerzo,
  nunca fatal).
- La construcción de la librería y las funciones de similitud se reutilizan
  BYTE-IDENTICAS del notebook del envío 3 (casmi_runtime_notebook.ipynb,
  celdas 2 y 3), ya validado en Kaggle (público 0.146).

v2 (2026-11-14, fix rueda RDKit): el envío v1 (ref 58333078, público
0.004) corrió sin RDKit porque la rueda cp312 no instaló en el contenedor
de Kaggle (Python 3.13, evidencia: log de la corrida, rutas
/usr/local/lib/python3.13/ en el stderr de nbconvert, líneas 159/161 de
kaggle_log_generativo_v1.txt). La v2 lleva las ruedas cp312 + cp313, las
prueba en orden (la de este Python primero) y loguea sys.version +
returncode + stderr de pip por intento. No cambia la lógica del modelo,
el umbral, la fusión ni el resto del notebook.

Rediseño v2 (2026-11-14, aplicación del rediseño generativo v2,
REDISENO_GENERATIVO_V2_2026-11-14.md, GO firme de Datos + Auditor):
blindaje de la fusión (componente 2: GENERATIVO_ACTIVO = HAS_RDKIT, celda 6
no genera sin RDKit, celda 7 trata gen_cands como vacío sin RDKit,
décimo check blindaje_coherente) y hook de validación del fallback
(componente 5: CASMI_VALIDACION_SIN_RDKIT=1, inerte en Kaggle). El umbral
0.95 sigue siendo el default; la re-calibración del portón (componente 3)
y la medición G4 (componente 4) se aplican por separado.

Uso:
  python3 verificacion/construir_notebook_generativo.py
  -> escribe verificacion/casmi_notebook_generativo_v2.ipynb
"""
import json
from pathlib import Path

import nbformat as nbf

REPO = Path("/home/hatch/workspace/enveda-casmi-2026")
NB_RUNTIME = REPO / "casmi_runtime_notebook.ipynb"
NB_OUT = REPO / "verificacion" / "casmi_notebook_generativo_v2.ipynb"


def celda_codigo_desde_runtime(indice, marcador):
    nb = json.loads(NB_RUNTIME.read_text())
    src = "".join(nb["cells"][indice]["source"])
    assert marcador in src, \
        f"celda {indice} del notebook runtime no trae {marcador}"
    return src


CODE_LIB = celda_codigo_desde_runtime(2, "def binear_fila")
CODE_SIM = celda_codigo_desde_runtime(3, "def score_query")

# --- Parche memoria CODE_LIB (2026-11-14, constructor v1 -> v2) ---
# Causa: la construcción CSR original (COO->CSR con copia, ~1-2 GB extra)
# mataba el proceso por OOM al final del build de la librería: dos runs
# locales (run A y run B) murieron sin traceback justo tras el chunk 26
# (RSS 5,521 MB, límite 6,144 MB). Reproducido con un test sintético del
# mismo tamaño (exit 137, oomKills 1). La equivalencia de la construcción
# sin copia está verificada en /tmp/test_csr_equiv.py (200k filas:
# indptr/indices/data idénticos, (X1 != X2).nnz = 0).
# Los datos ya vienen en orden canónico CSR (filas crecientes; dentro de
# cada fila, bins ordenados por np.unique y sin duplicados), así que el
# indptr se deriva de las longitudes por fila y la CSR se construye con
# copy=False. En Kaggle (13+ GB) el cambio es neutro; en esta caja es lo
# que permite validar. La lógica del loop no cambia.
def _parche_csr_sin_copia(src):
    a1 = "mol = np.empty(rows_upper, dtype=np.int32)"
    assert src.count(a1) == 1, "CODE_LIB: no se encontró la asignación de mol"
    src = src.replace(a1, a1 + "\nrowlen = np.empty(rows_upper, dtype=np.int32)", 1)
    a2 = "        n = len(bins)"
    assert src.count(a2) == 1, "CODE_LIB: no se encontró 'n = len(bins)'"
    src = src.replace(a2, a2 + "\n        rowlen[rowidx] = n", 1)
    a3 = ("X = sparse.csr_matrix((D[:off], (R[:off], C[:off])), shape=(rowidx, NBINS))\n"
          "X.sum_duplicates()")
    assert src.count(a3) == 1, "CODE_LIB: no se encontró la construcción CSR original"
    nuevo = ("# Construcción CSR sin copia: los datos ya vienen en orden canónico\n"
             "# (filas crecientes; bins ordenados por np.unique, sin duplicados\n"
             "# dentro de cada fila), así que el indptr sale de rowlen.\n"
             "# Evita el pico de ~1-2 GB de la conversión COO->CSR que en esta\n"
             "# caja (6 GB) mataba el proceso por OOM al final del build.\n"
             "indptr = np.zeros(rowidx + 1, dtype=np.int32)\n"
             "np.cumsum(rowlen[:rowidx], out=indptr[1:])\n"
             "X = sparse.csr_matrix((D[:off], C[:off], indptr), shape=(rowidx, NBINS), copy=False)\n"
             "X.has_canonical_format = True\n"
             "X.sum_duplicates()")
    src = src.replace(a3, nuevo, 1)
    return src


CODE_LIB = _parche_csr_sin_copia(CODE_LIB)

# --- Parche memoria CODE_LIB (2): liberar heap retenido tras el build ---
# La fragmentación del heap dejaba RSS ~5.5 GB tras el build; sin liberar,
# la copia de X[order] en la celda siguiente mataba el proceso por OOM.
CODE_LIB += (
    "\n# Liberar la memoria retenida por el allocator tras el build.\n"
    "import gc as _gc\n"
    "_gc.collect()\n"
    "try:\n"
    "    import ctypes as _ctypes\n"
    "    _ctypes.CDLL('libc.so.6').malloc_trim(0)\n"
    "except Exception:\n"
    "    pass\n"
)


def _parche_memoria_sim(src):
    # (a) Bloques de row_stats más chicos (200k -> 50k filas): los temporales
    # por bloque bajan de ~700 MB a ~175 MB.
    b1 = "    B = 200_000"
    assert src.count(b1) == 1, "CODE_SIM: no se encontró 'B = 200_000'"
    src = src.replace(b1, "    B = 50_000", 1)
    # (a2) No materializar data en float64 completo (+1,005 MB de pico):
    # el astype se hace por bloque. Bit-idéntico (float32->float64 exacto
    # por elemento; verificado en /tmp/test_rowstats_equiv.py). Sin esto,
    # row_stats mataba el proceso por OOM en esta caja (6 GB) aunque
    # X[order] ya no se sumara a su pico.
    b1b = "    data = X.data.astype(np.float64)"
    assert src.count(b1b) == 1, "CODE_SIM: no se encontró 'data = X.data.astype'"
    src = src.replace(b1b, "    data = X.data  # float32; el astype a float64 es por bloque", 1)
    b1c = "        d = data[sl]"
    assert src.count(b1c) == 1, "CODE_SIM: no se encontró 'd = data[sl]'"
    src = src.replace(b1c, "        d = data[sl].astype(np.float64)", 1)
    # (b) row_stats ANTES de X[order]: row_stats es por fila
    # (permutación-equivariante), así que se calcula sobre la librería sin
    # ordenar y luego se permutan los tres vectores. El pico de X[order]
    # (+1 GB) y el de row_stats ya no se suman; el orden original mataba el
    # proceso por OOM en esta caja (6 GB) al inicio de esta celda.
    # Verificado bit-idéntico en /tmp/test_rowstats_equiv.py (200k filas:
    # rowsum/S/norm_cos iguales, max diff 0.0).
    b2 = ("order = np.argsort(prec_lib, kind='stable')\n"
          "X = X[order]\n"
          "prec_s = prec_lib[order].astype(np.float64)\n"
          "mol_s = mol_lib[order]\n"
          "del order, prec_lib, mol_lib\n"
          "rowsum, S_row, norm_cos = row_stats(X)")
    assert src.count(b2) == 1, "CODE_SIM: no se encontró el bloque de ordenación"
    nuevo = ("order = np.argsort(prec_lib, kind='stable')\n"
             "# row_stats es por fila: se calcula sin ordenar y se permutan los\n"
             "# vectores (bit-idéntico, verificado). Así el pico de X[order]\n"
             "# no se suma al de row_stats (OOM en esta caja con el orden original).\n"
             "rowsum, S_row, norm_cos = row_stats(X)\n"
             "rowsum, S_row, norm_cos = rowsum[order], S_row[order], norm_cos[order]\n"
             "X = X[order]\n"
             "prec_s = prec_lib[order].astype(np.float64)\n"
             "mol_s = mol_lib[order]\n"
             "del order, prec_lib, mol_lib")
    src = src.replace(b2, nuevo, 1)
    return src


CODE_SIM = _parche_memoria_sim(CODE_SIM)

MD1 = """# CASMI 2026 — Envío generativo: híbrido retrieval + decodificador huella→SMILES

**Método (protocolo `PROTOCOLO_GENERATIVO_DE_NOVO.md` §§6/9, Fase 3):** híbrido retrieval-primero. Por cada molécula del `test.parquet` que el notebook reciba se calculan dos listas: (a) **retrieval baseline-1** — búsqueda en la librería espectral construida desde `train.parquet` (bins 0.01 Da, top-100 picos, precursor ±10 ppm, coseno GNPS), top-25 por InChIKey14; (b) **generativo familia B** — etapa 1 kNN espectro→huella Morgan 2048 (top-10 por coseno, media ponderada) + decodificador Transformer huella→SMILES (beam-25, checkpoint del piloto B re-entrenado). Si el top-1 coseno del retrieval es ≥ 0.95, el retrieval ocupa las 25 posiciones; si es menor, el generativo va primero y el retrieval rellena hasta 25. Dedup por InChIKey14 tautomer-canonicalizado (igual que el evaluador); si una molécula queda con menos de 25, se repite su mejor candidato (como el envío 3); "CCO" solo si no hay ningún candidato.

**Por qué en runtime:** Kaggle re-ejecuta el notebook y sustituye `test.parquet` por el test oculto. Este notebook genera las predicciones desde el test que reciba, así que funciona con el test oculto.

**Cumplimiento:** sin internet; datos de la competencia (`train.parquet` + `test.parquet`) + dataset privado con el checkpoint, el script del decodificador, el pool de huellas y la rueda RDKit (trabajo propio, piloto B); sin datos externos. El notebook debe correrse con **internet desactivado** y acelerador **GPU** (medido: beam-25 en CPU ~96 s/molécula, 10.7 h para 400, no cabe en 9 h; en GPU T4 el total estimado es ~15 min para el test visible)."""

CODE1 = """import hashlib, os, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

T0 = time.time()
def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time()-T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)

# v2: la versión de Python del contenedor de Kaggle nunca se imprimió en
# la v1 (el log no trae sys.version); sin ella no se puede saber qué rueda
# calza. Se imprime siempre, al inicio.
log(f"python: {sys.version.replace(chr(10), ' ')} | ejecutable: {sys.executable}")

# --- Parámetros del híbrido (protocolo §6, Fase 3) ---
# UMBRAL_COS: top-1 coseno de retrieval a partir del cual el retrieval ocupa
# las primeras posiciones (regla medida en el blend). Por debajo, el
# generativo va primero y el retrieval rellena. Se puede sobreescribir por
# entorno SOLO para validación local (en Kaggle la variable no existe).
UMBRAL_COS = float(os.environ.get("CASMI_UMBRAL_COS", "0.95"))
BEAM = 25
LOTE_BEAM = 32
UMBRAL_BIN = 0.5
TOPK_KNN = 10

BIN_W = 0.01
NBINS = 200_000
TOPK = 100
PMZ_MIN, PMZ_MAX = 100.0, 2000.0
NPEAK_MIN = 3
# CHUNK 25k (antes 100k, valor del runtime): el DataFrame de 100k filas con
# las listas ms2 (objetos Python) costaba ~600-700 MB transitorios por chunk
# y el build llegaba a RSSmax ~5,514 MB en esta caja (límite 6,144 MB) —
# el run E murió por OOM en el chunk 22 sin haber tocado la celda 3.
# Con 25k el pico por chunk baja a ~150-175 MB y el build queda ~500 MB
# por debajo del pico anterior. No cambia el contenido de la librería
# (el procesamiento es por fila, independiente del tamaño de chunk):
# el run F debe dar las mismas 2,438,064 filas y 125,646,101 nnz que los
# runs C/D. En Kaggle (13+ GB) el cambio es neutro.
CHUNK = 25_000

def resolver_datos():
    # Input del concurso en Kaggle: /kaggle/input/<slug>/ con train+test.
    kagg = Path('/kaggle/input')
    if kagg.is_dir():
        for d in sorted(kagg.iterdir()):
            if d.is_dir() and (d / 'train.parquet').exists() and (d / 'test.parquet').exists():
                return d
        for p in sorted(kagg.rglob('train.parquet')):
            if (p.parent / 'test.parquet').exists():
                return p.parent
    for d in [Path('datos'),
              Path('/home/hatch/workspace/enveda-casmi-2026/datos')]:
        if (d / 'train.parquet').exists() and (d / 'test.parquet').exists():
            return d
    raise FileNotFoundError('no se encontró train.parquet + test.parquet')

ARCHIVOS_GEN = {
    'script': 'entrenar_decodificador.py',
    'checkpoint': 'modelo.pt',
    'huellas_pool': 'huellas_pool.npz',
    'mol_index': 'mol_index.npy',
}
# v2: las ruedas RDKit ya no son un archivo opcional fijo (la v1 llevaba
# solo la cp312). Se descubren por glob en _buscar_ruedas() más abajo:
# cp312 + cp313 de RDKit 2026.03.3; se prueban en orden, la de este
# Python primero.
LOCAL_GEN = [
    Path('/home/hatch/workspace/casmi-piloto-b-resultados'),
    Path('/home/hatch/workspace/casmi-piloto-b-resultados/kaggle_dataset'),
    Path('/home/hatch/workspace/enveda-casmi-2026/verificacion/piloto_b/v2-fix-beam'),
    Path('/home/hatch/workspace/enveda-casmi-2026/intermedios'),
    Path('/home/hatch/workspace/enveda-casmi-2026/verificacion'),
    Path('verificacion/piloto_b/v2-fix-beam'),
    Path('intermedios'),
    Path('verificacion'),
]

def _buscar(nombre):
    kagg = Path('/kaggle/input')
    if kagg.is_dir():
        hits = sorted(kagg.rglob(nombre))
        if hits:
            return hits[0]
    for d in LOCAL_GEN:
        p = d / nombre
        if p.exists():
            return p
    return None

def _md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

GEN = {}
for _clave, _nombre in ARCHIVOS_GEN.items():
    _p = _buscar(_nombre)
    if _p is None:
        raise FileNotFoundError(f'artefacto generativo no encontrado: {_nombre}')
    GEN[_clave] = _p
    log(f"gen {_clave}: {_p} ({_p.stat().st_size:,} B, md5 {_md5(_p)})")
# --- Ruedas RDKit del dataset (v2: cp312 + cp313) ---
# La v1 llevaba solo la cp312 y el contenedor de Kaggle es Python 3.13
# (evidencia: log de la corrida v1, rutas /usr/local/lib/python3.13/ en
# el stderr de nbconvert, líneas 159/161 de kaggle_log_generativo_v1.txt).
# Se prueban en orden: primero la que calza con este Python, luego las
# demás; cada intento loguea returncode + stdout/stderr de pip.
def _buscar_ruedas():
    _hits = []
    _kagg = Path('/kaggle/input')
    if _kagg.is_dir():
        _hits.extend(sorted(_kagg.rglob('rdkit-2026.3.3-*.whl')))
    for _d in LOCAL_GEN:
        _hits.extend(sorted(_d.glob('rdkit-2026.3.3-*.whl')))
    _vistos, _unicas = set(), []
    for _p in _hits:
        if _p.name not in _vistos:
            _vistos.add(_p.name)
            _unicas.append(_p)
    _tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    _unicas.sort(key=lambda p: (0 if _tag in p.name else 1, p.name))
    return _unicas

RUEDAS_RDKIT = _buscar_ruedas()
log(f"ruedas RDKit en el dataset: {[p.name for p in RUEDAS_RDKIT] or 'NINGUNA'}")

DATOS = resolver_datos()
WORK = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('verificacion/generativo_check')
WORK.mkdir(parents=True, exist_ok=True)
log(f"datos: {DATOS} | trabajo: {WORK} | umbral coseno: {UMBRAL_COS} | beam: {BEAM}")
log(f"train: {(DATOS / 'train.parquet').stat().st_size / 1e9:.2f} GB | "
    f"test: {(DATOS / 'test.parquet').stat().st_size / 1e6:.1f} MB")

# --- RDKit (mejor esfuerzo, nunca fatal) ---
# El scorer canonicaliza con RDKit 2026.03.3 (TautomerEnumerator +
# MolToInchiKey, primer bloque). Si la imagen de Kaggle no lo trae, se
# instala offline desde una rueda del dataset (v2: se prueban cp312 y
# cp313 en orden, la de este Python primero; cada intento loguea el
# returncode y el stderr de pip). Si ninguna instala, el notebook sigue
# con dedup por SMILES exacto y sin filtro de validez.
HAS_RDKIT = False
Chem = None
_TAUT = None
# Hook de validación del blindaje (componente 5, rediseño v2 2026-11-14):
# con CASMI_VALIDACION_SIN_RDKIT=1 el notebook NO importa RDKit aunque esté
# presente ni intenta instalar la rueda; el generativo queda DESACTIVADO.
# En Kaggle la variable no existe: comportamiento normal.
SIN_RDKIT_FORZADO = os.environ.get("CASMI_VALIDACION_SIN_RDKIT", "0") == "1"
if SIN_RDKIT_FORZADO:
    RUEDAS_RDKIT = []
    log("validación sin RDKit forzada (hook CASMI_VALIDACION_SIN_RDKIT=1): "
        "no intento instalar la rueda")
try:
    if SIN_RDKIT_FORZADO:
        raise ImportError("hook CASMI_VALIDACION_SIN_RDKIT=1")
    from rdkit import Chem as _Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize as _rms
    Chem = _Chem
    _TAUT = _rms.TautomerEnumerator()
    HAS_RDKIT = True
    log('RDKit ya disponible en el entorno')
except Exception:
    for _rueda in RUEDAS_RDKIT:
        log(f"RDKit no presente; intento instalar la rueda offline: {_rueda.name}")
        try:
            _r = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index',
                 str(_rueda)],
                capture_output=True, text=True, timeout=600)
            log(f"pip install {_rueda.name}: returncode={_r.returncode}")
            if _r.stdout.strip():
                log(f"pip stdout ({_rueda.name}): {_r.stdout.strip()[-2000:]}")
            if _r.stderr.strip():
                log(f"pip stderr ({_rueda.name}): {_r.stderr.strip()[-2000:]}")
            from rdkit import Chem as _Chem
            from rdkit.Chem.MolStandardize import rdMolStandardize as _rms
            Chem = _Chem
            _TAUT = _rms.TautomerEnumerator()
            HAS_RDKIT = True
            log(f"RDKit instalado offline desde la rueda {_rueda.name}")
            break
        except Exception as e:
            log(f"rueda {_rueda.name} falló ({type(e).__name__}: {e}); pruebo la siguiente")
    if not HAS_RDKIT:
        if SIN_RDKIT_FORZADO:
            log("RDKit DESACTIVADO por el hook de validación "
                "(CASMI_VALIDACION_SIN_RDKIT=1); sigo sin RDKit")
        else:
            log("RDKit NO disponible (ninguna rueda instaló); sigo sin RDKit "
                "(dedup por SMILES exacto, sin filtro de validez)")

# --- Blindaje v2 (componente 2, rediseño 2026-11-14) ---
# El generativo solo actúa si hay RDKit para filtrar sus SMILES inválidos
# (dedup por InChIKey14 + filtro de validez). Sin RDKit el notebook es el
# envío 3 (retrieval puro): no genera ni fusiona candidatos generativos.
# La v1 (ref 58333078, público 0.004) falló porque sin RDKit el generativo
# sin filtrar desplazó al retrieval en el test oculto.
GENERATIVO_ACTIVO = HAS_RDKIT
log(f"generativo: {'ACTIVO' if GENERATIVO_ACTIVO else 'DESACTIVADO (blindaje; retrieval puro)'}")

import torch
DISPOSITIVO = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(min(4, os.cpu_count() or 1))
log(f"torch {torch.__version__} | dispositivo {DISPOSITIVO} | "
    f"hilos {torch.get_num_threads()} | RDKit: {HAS_RDKIT}")
"""

CODE4 = """# ---- Predicción retrieval (baseline 1, 03_predecir_test.py) ----
# Por molecule_id: top-25 por InChIKey14 (máxima similitud por espectro) y
# top-1 coseno (confianza del retrieval, regla del híbrido §6).
test = pd.read_parquet(DATOS / 'test.parquet')
log(f"test recibido: {len(test):,} espectros, {test['molecule_id'].nunique():,} moléculas")
SUBSET = int(os.environ.get('CASMI_VALIDACION_SUBSET', '0'))
if SUBSET > 0:
    _mids_orden = sorted(test['molecule_id'].unique().tolist())[:SUBSET]
    test = test[test['molecule_id'].isin(_mids_orden)].copy()
    log(f"VALIDACIÓN LOCAL: subconjunto de {SUBSET} moléculas "
        f"({len(test):,} espectros) — hook CASMI_VALIDACION_SUBSET")

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

order = np.argsort(q_mol_id, kind='stable')
q_mol_s = q_mol_id[order]
uniq, starts = np.unique(q_mol_s, return_index=True)
bounds = list(starts) + [len(q_mol_s)]
todos = [str(m) for m in uniq.tolist()]
ret_top25 = {}
ret_top1_cos = {}
t0 = time.time()
for ui, mid in enumerate(uniq):
    s, e = bounds[ui], bounds[ui + 1]
    best, mols_c = [], []
    top1 = 0.0
    for j in order[s:e]:
        m, c, _en = score_query(X, prec_s, mol_s, rowsum, S_row, norm_cos,
                                Q[j], float(q_pmz[j]))
        if m is None:
            continue
        best.append(c)
        mols_c.append(m)
        top1 = max(top1, float(c.max()))
    if not best:
        ret_top25[str(mid)] = []
        ret_top1_cos[str(mid)] = 0.0
        continue
    top = rank_molecule(best, mols_c)[:25]
    lista = []
    for code in top:
        ik = str(mols[code])
        sm = rep.get(ik)
        if sm:
            lista.append((ik, sm))
    ret_top25[str(mid)] = lista
    ret_top1_cos[str(mid)] = top1
    if (ui + 1) % 100 == 0:
        log(f"retrieval: {ui + 1:,}/{len(uniq):,} moléculas ({time.time() - t0:.0f}s)")
_fuerte = sum(1 for m in todos if ret_top1_cos[m] >= UMBRAL_COS)
log(f"retrieval completo: {len(todos):,} moléculas | top1_cos media "
    f"{np.mean([ret_top1_cos[m] for m in todos]):.4f} | "
    f"con top1_cos >= {UMBRAL_COS}: {_fuerte} "
    f"({100.0 * _fuerte / max(1, len(todos)):.1f}%)")
"""

CODE5 = """# ---- Etapa 1 kNN: espectro -> huella Morgan (09_predecir_test_paso2.py) ----
# Por espectro: top-10 moléculas por coseno en la librería, media ponderada
# de sus huellas del pool. Por molécula: media de sus espectros.
# La huella predicha se binariza a 0.5 (medido en
# verificacion/medir_binarizacion_etapa1.py: Tanimoto 0.4855, 24 bits/fila,
# dentro de la distribución de entrenamiento del decodificador).
pool = sparse.load_npz(GEN['huellas_pool']).tocsr().astype(np.float32)
pool_nbits = np.diff(pool.indptr).astype(np.float32)
log(f"pool huellas: {pool.shape[0]:,} x {pool.shape[1]:,}, nnz {pool.nnz:,}")
_mi = np.load(GEN['mol_index'], allow_pickle=True)
_mi = np.array([str(x) for x in _mi])
assert len(_mi) == len(mols) and bool((_mi == mols).all()), \
    'mol_index no coincide con mols de train (pool desalineado)'
log('pool alineado con mols (mol_index == train inchikey14 ordenados)')
del _mi

P_spec = np.zeros((Q.shape[0], 2048), dtype=np.float32)
n_sin = 0
t0 = time.time()
for j in range(Q.shape[0]):
    m, c, _en = score_query(X, prec_s, mol_s, rowsum, S_row, norm_cos,
                            Q[j], float(q_pmz[j]))
    if m is None:
        n_sin += 1
        continue
    best = {}
    for code, cc in zip(m.tolist(), c.tolist()):
        if cc > best.get(code, -1.0):
            best[code] = float(cc)
    top = sorted(best.items(), key=lambda kv: -kv[1])[:TOPK_KNN]
    w = np.clip(np.asarray([cc for _, cc in top], dtype=np.float32), 0.0, None)
    if w.sum() <= 0:
        n_sin += 1
        continue
    w /= w.sum()
    codes = np.asarray([cc for cc, _ in top], dtype=np.int32)
    F = pool[codes].toarray().astype(np.float32)
    P_spec[j] = w @ F
    if (j + 1) % 500 == 0:
        log(f"etapa1 kNN: {j + 1:,}/{Q.shape[0]:,} espectros ({time.time() - t0:.0f}s)")
log(f"P_spec kNN: {P_spec.shape}, espectros sin candidatos: {n_sin}")

# Agregar por molecule_id (media de espectros; igual que 09_predecir_test_paso2.py).
counts = np.diff(bounds).astype(np.float32)
rows_a = np.repeat(np.arange(len(uniq)), counts.astype(np.int64))
cols_a = order.astype(np.int32)
data_a = np.repeat((1.0 / counts).astype(np.float32), counts.astype(np.int64))
A = sparse.csr_matrix((data_a, (rows_a, cols_a)), shape=(len(uniq), len(q_mol_id)))
P_test = np.asarray(A @ P_spec, dtype=np.float32)
del P_spec, A
pos_mol = {mid: i for i, mid in enumerate(todos)}
B_test = (P_test > UMBRAL_BIN).astype(np.float32)
log(f"P_test por molécula: {P_test.shape}, bits>0.5/fila media {B_test.sum(1).mean():.1f}, "
    f"filas sin bits: {int((B_test.sum(1) == 0).sum())}")

# Cordura: Tanimoto continuo de P_test vs huella del top-1 del retrieval.
_top1_codes = np.array([mol_index[ret_top25[m][0][0]] for m in todos
                        if ret_top25[m]], dtype=np.int64)
if len(_top1_codes):
    _Y = pool[_top1_codes].toarray().astype(np.float32)
    _P = np.clip(P_test[[pos_mol[m] for m in todos if ret_top25[m]]], 0.0, 1.0)
    _dots = np.einsum('ij,ij->i', _P, _Y)
    _den = (np.einsum('ij,ij->i', _P, _P) + _Y.sum(1) - _dots)
    _t = np.zeros(len(_P), dtype=np.float32)
    _m = _den > 0
    _t[_m] = _dots[_m] / _den[_m]
    log(f"cordura etapa 1: Tanimoto continuo P_test vs top-1 retrieval: "
        f"media {float(_t.mean()):.4f}, mediana {float(np.median(_t)):.4f} "
        f"(visible clase-1 ~0.62; en oculto novel se espera ~0.31, medido paso 2)")
"""

CODE6 = """# ---- Etapa 2: decodificador huella -> SMILES (beam-25, script fixeado) ----
# Solo se genera para las moléculas que lo necesitan (top1_cos < UMBRAL_COS).
# El checkpoint es el re-entrenado (Colab 2026-11-13 20:45, beam exact-match
# 0.2525, greedy 0.2335). La entrada es la huella BINARIA (umbral 0.5).
sys.path.insert(0, str(GEN['script'].parent))
import entrenar_decodificador as dec

ckpt = torch.load(str(GEN['checkpoint']), map_location=DISPOSITIVO,
                  weights_only=False)
config = ckpt['config']
tok = dec.Tokenizador(ckpt['vocab'])
modelo = dec.construir_modelo(len(ckpt['vocab']), d_model=config['d_model'],
                              nhead=config['nhead'],
                              num_capas=config['num_capas'],
                              dim_ff=config['dim_ff'],
                              dropout=config['dropout'],
                              k_memoria=config['k_memoria'],
                              max_len=config['max_len'])
modelo.load_state_dict(ckpt['state_dict'])
modelo.to(DISPOSITIVO)
modelo.eval()
log(f"checkpoint cargado: {GEN['checkpoint'].name} "
    f"(n_train {config['n_train']}, épocas {config['epocas']}, "
    f"max_len {config['max_len']}, vocab {len(ckpt['vocab'])}, "
    f"parámetros {sum(p.numel() for p in modelo.parameters()):,})")

gen_mols = ([mid for mid in todos if ret_top1_cos[mid] < UMBRAL_COS]
            if GENERATIVO_ACTIVO else [])
if not GENERATIVO_ACTIVO:
    log("generativo DESACTIVADO (blindaje): no se genera ninguna molécula")
log(f"moléculas que necesitan generación (top1_cos < {UMBRAL_COS}): "
    f"{len(gen_mols)}/{len(todos)}")
gen_cands = {}
t_gen0 = time.time()
for ini in range(0, len(gen_mols), LOTE_BEAM):
    lote = gen_mols[ini:ini + LOTE_BEAM]
    idxs = [pos_mol[mid] for mid in lote]
    huella_t = torch.tensor(B_test[idxs], device=DISPOSITIVO)
    listas, _scores = dec.generar_beam(modelo, tok, huella_t, DISPOSITIVO,
                                       config['max_len'], beam=BEAM)
    for mid, cands in zip(lote, listas):
        gen_cands[mid] = list(cands)
    log(f"beam: {min(ini + LOTE_BEAM, len(gen_mols)):,}/{len(gen_mols):,} "
        f"moléculas ({time.time() - t_gen0:.0f}s)")
if gen_cands and HAS_RDKIT:
    _tot = sum(len(v) for v in gen_cands.values())
    _val = sum(1 for v in gen_cands.values() for s in v
               if s and Chem.MolFromSmiles(s) is not None)
    log(f"validez beam (RDKit): {_val}/{_tot} = {_val / max(1, _tot):.4f} "
        f"(GEN-VAL medido 0.7463)")
"""

CODE7 = """# ---- Híbrido: fusión retrieval + generativo -> submission.csv ----
# Clave de dedup: InChIKey14 tautomer-canonicalizado (igual que el evaluador
# 30_evaluar_generativo.py) si hay RDKit; si no, el SMILES exacto.
# Los SMILES inválidos del generativo se descartan (solo si hay RDKit para
# detectarlos); los del retrieval vienen de train y son válidos.
def clave_candidata(smiles):
    if not HAS_RDKIT:
        return smiles if smiles else None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        m2 = _TAUT.Canonicalize(m)
    except Exception:
        m2 = m
    try:
        return Chem.MolToInchiKey(m2).split('-')[0]
    except Exception:
        return None

def agregar(out, seen, smiles, ik=None):
    # Rechazar cadenas vacías ANTES de la clave: en RDKit 2026.03.3
    # MolFromSmiles('') devuelve un Mol vacío (no None) y MolToInchiKey es
    # '', así que el '' pasaba como candidato "válido" con clave ''. El run F
    # dejó 2 candidatos vacíos (m_006153 pos 5, m_00b5aa pos 8) que la
    # verificación no cazó (sin_vacios solo miraba la celda completa).
    # El '' no es un SMILES: no puede puntuar y podría romper el scorer.
    if not smiles or not str(smiles).strip():
        return False
    k = ik if ik is not None else clave_candidata(smiles)
    if k is None or k == '' or k in seen:
        return False
    seen.add(k)
    out.append(smiles)
    return True

out_rows = []
n_ret_only = 0
n_gen_first = 0
n_cco = 0
for mid in todos:
    ret = ret_top25.get(mid, [])
    # Defensa en profundidad (blindaje, componente 2): sin RDKit los
    # candidatos generativos se tratan como vacíos aunque existieran.
    gen = gen_cands.get(mid, []) if GENERATIVO_ACTIVO else []
    out = []
    seen = set()
    if ret_top1_cos.get(mid, 0.0) >= UMBRAL_COS:
        for ik, sm in ret:
            agregar(out, seen, sm, ik)
        n_ret_only += 1
    else:
        for sm in gen:
            agregar(out, seen, sm)
        for ik, sm in ret:
            agregar(out, seen, sm, ik)
        # Solo cuenta como generativo-primero si el generativo estaba
        # activo: con el blindaje (sin RDKit) ninguna fila es
        # generativo-primero aunque tome esta rama.
        if GENERATIVO_ACTIVO:
            n_gen_first += 1
    while len(out) < 25:
        out.append(out[0] if out else 'CCO')
    if len(out) == 25 and out[0] == 'CCO' and len(set(out)) == 1:
        n_cco += 1
    out_rows.append({'molecule_id': mid, 'smiles': ';'.join(out[:25])})
sub = pd.DataFrame(out_rows)
sub.to_csv(WORK / 'submission.csv', index=False)
_n_unicos = sub['smiles'].apply(lambda s: len(set(s.split(';'))))
log(f"submission escrita: {len(sub)} filas | retrieval-primero {n_ret_only} | "
    f"generativo-primero {n_gen_first} | filas solo-CCO {n_cco} | "
    f"candidatos únicos/fila media {_n_unicos.mean():.1f}")
"""

CODE8 = """# ---- Verificación de la salida (genérica: vale para el test público y el oculto) ----
sub2 = pd.read_csv(WORK / 'submission.csv')
test_ids = set(test['molecule_id'].unique().tolist())
n_c = sub2['smiles'].apply(lambda s: len(str(s).split(';')))
n_cco_filas = int((sub2['smiles'] == ';'.join(['CCO'] * 25)).sum())
checks = {
    'columnas_exactas': list(sub2.columns) == ['molecule_id', 'smiles'],
    'filas_igual_a_moleculas_test': len(sub2) == len(test_ids),
    'ids_unicos': bool(sub2['molecule_id'].is_unique),
    'ids_iguales_a_test': set(sub2['molecule_id'].tolist()) == test_ids,
    'sin_nulos': bool(sub2['smiles'].notna().all() and sub2['molecule_id'].notna().all()),
    'sin_vacios': bool((sub2['smiles'].astype(str).str.strip() != '').all()),
    # El run F pasó sin_vacios con 2 candidatos vacíos dentro de la lista
    # (la celda completa no estaba vacía). Este check mira cada candidato.
    'sin_candidatos_vacios': bool(all(c.strip() != '' for s in sub2['smiles'] for c in str(s).split(';'))),
    'exactamente_25_por_fila': bool((n_c == 25).all()),
    'cero_filas_solo_CCO': n_cco_filas == 0,
}
# Décimo check (blindaje, componente 2, rediseño v2): el número de filas
# generativo-primero debe coincidir con el esperado por el portón.
# Sin RDKit (blindaje) el esperado es 0: ninguna fila va al generativo.
n_esperado = (sum(1 for m in todos if ret_top1_cos.get(m, 0.0) < UMBRAL_COS)
              if GENERATIVO_ACTIVO else 0)
blindaje_coherente = (n_gen_first == n_esperado)
checks['blindaje_coherente'] = bool(blindaje_coherente)
md5 = hashlib.md5(open(WORK / 'submission.csv', 'rb').read()).hexdigest()
for k, v in checks.items():
    print(('OK   ' if v else 'FAIL ') + f'- {k}')
print(f"filas: {len(sub2)} | candidatos totales: {int(n_c.sum())} | "
      f"min/fila {int(n_c.min())} max/fila {int(n_c.max())} | "
      f"filas solo-CCO: {n_cco_filas} | retrieval-primero: {n_ret_only} | "
      f"generativo-primero: {n_gen_first} (esperado: {n_esperado}) | "
      f"generativo: {'ACTIVO' if GENERATIVO_ACTIVO else 'DESACTIVADO'} | "
      f"RDKit: {HAS_RDKIT} | dispositivo: {DISPOSITIVO}")
print(f"md5 submission.csv: {md5}")
print(f"tiempo total: {time.time() - T0:.1f}s")
assert all(checks.values()), 'verificación final con fallos'
print('VERIFICACIÓN FINAL: TODO OK')
"""

MD2 = """## Notas de cumplimiento y de ejecución

- **Sin internet:** el notebook no usa red en ningún punto; la configuración de internet del notebook debe quedar desactivada al enviar (requisito del concurso). La única instalación es `pip install --no-index` desde una rueda RDKit del dataset (offline; v2: ruedas cp312 + cp313 de RDKit 2026.03.3, se prueban en orden y se loguea `sys.version` + returncode + stderr de pip por intento); si ninguna instala, el notebook sigue sin RDKit.
- **Datos:** `train.parquet` y `test.parquet` del input del concurso + dataset privado con el checkpoint del piloto B (trabajo propio, re-entrenado en Colab 2026-11-13), el script fixeado del decodificador, el pool de huellas y `mol_index.npy` (derivados de `train.parquet`), y la rueda RDKit 2026.03.3 (misma versión que el scorer).
- **Acelerador:** GPU obligatoria (P100/T4). Medido en local (CPU, 4 hilos): beam-25 a ~96 s/molécula, así que 400 moléculas tardarían ~10.7 h en CPU y NO caben en el límite de 9 h. En GPU (T4, medido en Colab: 2,000 moléculas en 867 s) el total estimado para el test visible es ~15 min. CPU solo sirve para pruebas con pocas moléculas.
- **Híbrido:** retrieval-primero con umbral top-1 coseno 0.95 (protocolo §6). En el test visible todas las moléculas son clase-1 (top1_cos ~1.0), así que el generativo no dispara ahí; en el oculto (~17% clase-1 estimado) dispara en ~83%. La puerta G0 (formato) se verifica en el visible; la calidad del generativo con huella predicha (G3/G4) se mide en GEN-VAL con el evaluador, no en este notebook.
- **Relleno a 25:** cada fila trae exactamente 25 candidatos; si una rama da menos de 25 únicos, se completa con la otra rama y, en última instancia, repitiendo el mejor candidato (el sample del concurso trae 25 "CCO" idénticos por fila, así que los duplicados son aceptados por el formato). "CCO" solo si la molécula no tiene ningún candidato en ninguna rama.
- **Blindaje (rediseño v2, 2026-11-14):** el generativo solo actúa si hay RDKit (`GENERATIVO_ACTIVO = HAS_RDKIT`). Sin RDKit el notebook es el envío 3 (retrieval puro): no genera, no fusiona candidatos generativos, y el décimo check `blindaje_coherente` verifica que el número de filas generativo-primero coincida con el esperado por el portón (0 sin RDKit). La v1 (público 0.004) falló porque sin RDKit el generativo sin filtrar desplazó al retrieval en el test oculto.
- **Hook de validación (solo local):** `CASMI_VALIDACION_SIN_RDKIT=1` fuerza el modo sin RDKit (no importa RDKit aunque esté presente, no intenta instalar la rueda) para verificar el blindaje. En Kaggle la variable no existe: comportamiento normal.
- **Salida:** `submission.csv` en el working dir, una fila por cada `molecule_id` del test recibido, en orden alfabético de `molecule_id`."""

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(MD1),
    nbf.v4.new_code_cell(CODE1),
    nbf.v4.new_code_cell(CODE_LIB),
    nbf.v4.new_code_cell(CODE_SIM),
    nbf.v4.new_code_cell(CODE4),
    nbf.v4.new_code_cell(CODE5),
    nbf.v4.new_code_cell(CODE6),
    nbf.v4.new_code_cell(CODE7),
    nbf.v4.new_code_cell(CODE8),
    nbf.v4.new_markdown_cell(MD2),
]
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.12.3"}
nbf.write(nb, NB_OUT)
print(f"notebook escrito: {NB_OUT} ({len(nb.cells)} celdas, "
      f"{sum(1 for c in nb.cells if c.cell_type == 'code')} de código)")
print(f"CODE_LIB reutilizada del runtime: {len(CODE_LIB):,} caracteres "
      f"(binear_fila + build librería)")
print(f"CODE_SIM reutilizada del runtime: {len(CODE_SIM):,} caracteres "
      f"(score_query + rank_molecule)")
