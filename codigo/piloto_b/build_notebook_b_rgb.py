#!/usr/bin/env python3
"""Build the experimental Kaggle RGB -> Pilot B submission notebook."""

import ast
import json
from pathlib import Path

PILOTO = Path(__file__).resolve().parent
CODIGO = PILOTO.parent


def extraer_definiciones(path, nombres):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    encontrados = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node_names = {node.name}
        elif isinstance(node, ast.Assign):
            node_names = {
                target.id for target in node.targets
                if isinstance(target, ast.Name)
            }
        else:
            continue
        if node_names & nombres:
            encontrados.append((node.lineno, ast.get_source_segment(source, node)))
    encontrados.sort()
    faltantes = nombres - {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    } - {
        target.id for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    }
    if faltantes:
        raise ValueError(f"faltan definiciones en {path}: {sorted(faltantes)}")
    return "\n\n".join(source for _, source in encontrados)


DECODER_NAMES = {
    "NBITS", "SMILES_TOKEN_REGEX", "TOKEN_PAD", "TOKEN_BOS", "TOKEN_EOS",
    "TOKEN_UNK", "TOKENS_ESPECIALES", "tokenizar_smiles", "Tokenizador",
    "_torch", "construir_modelo", "mascara_causal", "vector_denso",
    "binarizar_predicciones_rgb", "generar_beam",
}
RGB_NAMES = {"l2_por_fila", "transformar_rgb"}
DECODER_DEFS = extraer_definiciones(
    PILOTO / "entrenar_decodificador.py", DECODER_NAMES
)
RGB_DEFS = extraer_definiciones(CODIGO / "21_entrenar_lineal_rgb.py",
                                RGB_NAMES)

CELDAS = [
    ("markdown", """# CASMI 2026 — RGB → Pilot B (submission experimental)

Este notebook aplica el predictor RGB de espectro→huella y el checkpoint
Pilot B afinado con predicciones **out-of-fold** para generar 25 SMILES por
`molecule_id` del `test.parquet` que Kaggle entregue. Es una ruta experimental:
en 2.000 consultas GEN-VAL, `MRR_novel` fue 0.001504 (IC95% pareado de la
diferencia contra el baseline incluye cero), por debajo del umbral propuesto
de 0.01. No reemplaza el notebook RGB validado.

## Inputs requeridos

1. El input de la competencia, con `test.parquet` (y opcionalmente
   `sample_submission.csv`).
2. El dataset RGB privado, con `W_lineal_rgb20k.npy` y `bins_rgb.npy`.
3. Un input adjunto que contenga el checkpoint **fine-tuned OOF** llamado
   `modelo.pt`. El notebook lo valida por su configuración (`p_rgb_real=0.5`,
   `rgb_top_k=100`) y falla explícitamente si se adjunta el checkpoint base.

Adjunta GPU para acelerar beam search. Internet no es necesario ni se usa;
PyTorch, NumPy, pandas, SciPy y PyArrow deben estar disponibles en el runtime
Kaggle. La salida se escribe en `/kaggle/working/submission.csv`.
"""),
    ("code", """import csv
import hashlib
import os
import re
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy import sparse

T0 = time.time()
INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("working")
WORK.mkdir(parents=True, exist_ok=True)

# Opcionales para probar fuera de Kaggle. En Kaggle se descubren bajo /kaggle/input.
COMPETITION_OVERRIDE = os.environ.get("CASMI_COMPETITION_DIR")
RGB_OVERRIDE = os.environ.get("CASMI_RGB_DIR")
CHECKPOINT_OVERRIDE = os.environ.get("CASMI_B_CHECKPOINT")

def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time() - T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)

def resolver_directorio(nombre_archivo, override=None):
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            path = path.parent
        if not (path / nombre_archivo).is_file():
            raise FileNotFoundError(f"{path} no contiene {nombre_archivo}")
        return path
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"no existe {INPUT_ROOT}; adjunta los inputs de Kaggle")
    encontrados = sorted({
        path.parent for path in INPUT_ROOT.rglob(nombre_archivo) if path.is_file()
    })
    if len(encontrados) != 1:
        raise FileNotFoundError(
            f"se esperaba exactamente un {nombre_archivo} bajo {INPUT_ROOT}; "
            f"encontrados: {encontrados}. Define el override correspondiente."
        )
    return encontrados[0]

COMP = resolver_directorio("test.parquet", COMPETITION_OVERRIDE)
RGB = resolver_directorio("W_lineal_rgb20k.npy", RGB_OVERRIDE)
if CHECKPOINT_OVERRIDE:
    CHECKPOINT = Path(CHECKPOINT_OVERRIDE).expanduser()
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"no existe checkpoint: {CHECKPOINT}")
else:
    checkpoints = sorted(
        path for path in INPUT_ROOT.rglob("modelo.pt") if path.is_file()
    ) if INPUT_ROOT.is_dir() else []
    if len(checkpoints) != 1:
        raise FileNotFoundError(
            f"se esperaba exactamente un modelo.pt bajo {INPUT_ROOT}; "
            f"encontrados: {checkpoints}. Define CASMI_B_CHECKPOINT."
        )
    CHECKPOINT = checkpoints[0]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("AVISO: no hay GPU; beam search puede tardar mucho.")
else:
    log(f"GPU: {torch.cuda.get_device_name(0)}")
log(f"competencia: {COMP} | RGB: {RGB} | checkpoint: {CHECKPOINT}")
log(f"test: {((COMP / 'test.parquet').stat().st_size) / 1e6:.1f} MB")

checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
config = checkpoint["config"]
if float(config.get("p_rgb_real", 0.0)) < 0.5 or config.get("rgb_top_k") != 100:
    raise ValueError(
        "se requiere el checkpoint OOF afinado con p_rgb_real>=0.5 y "
        f"rgb_top_k=100; config recibida: "
        f"p_rgb_real={config.get('p_rgb_real')}, rgb_top_k={config.get('rgb_top_k')}"
    )
log(f"checkpoint OOF validado: n_train={config.get('n_train')}, "
    f"épocas fine-tune={config.get('epocas')}, "
    f"p_rgb_real={config['p_rgb_real']}")
"""),
    ("code", "# Definiciones de tokenizador, modelo y beam search del evaluador B.\n\n"
     + DECODER_DEFS),
    ("code", "# Transformación RGB compartida con 21_entrenar_lineal_rgb.py.\n\n"
     + RGB_DEFS),
    ("code", """# Binea cada espectro igual que el runtime RGB, predice scores RGB
# y promedia las salidas por molécula antes de binarizar top-100.
BIN_W = 0.01
NBINS = 200_000
TOPK_PICOS = 100
PMZ_MIN, PMZ_MAX = 100.0, 2000.0

def binear_fila(mzs, intensidades):
    mzs = np.asarray(mzs, dtype=np.float64)
    intensidades = np.asarray(intensidades, dtype=np.float64)
    ok = (
        (intensidades > 0) & np.isfinite(mzs) & np.isfinite(intensidades)
        & (mzs < 2000.0) & (mzs >= 0)
    )
    if not ok.any():
        return None
    mzs, intensidades = mzs[ok], intensidades[ok]
    if len(intensidades) > TOPK_PICOS:
        idx = np.argpartition(-intensidades, TOPK_PICOS)[:TOPK_PICOS]
        mzs, intensidades = mzs[idx], intensidades[idx]
    bins = np.floor(mzs / BIN_W + 1e-9).astype(np.int32)
    bins = np.clip(bins, 0, NBINS - 1)
    ub, inv = np.unique(bins, return_inverse=True)
    vals = np.bincount(inv, weights=intensidades, minlength=len(ub))
    return ub, vals.astype(np.float32)

ids_table = pd.read_parquet(COMP / "test.parquet", columns=["molecule_id"])
if ids_table["molecule_id"].isna().any():
    raise ValueError("test.parquet contiene molecule_id nulos")
test_ids = ids_table["molecule_id"].drop_duplicates().tolist()
del ids_table
parquet = pq.ParquetFile(COMP / "test.parquet")
log(f"test recibido: {parquet.metadata.num_rows:,} espectros, "
    f"{len(test_ids):,} moléculas")

W = np.load(RGB / "W_lineal_rgb20k.npy", mmap_mode="r")
bins_rgb = np.load(RGB / "bins_rgb.npy")
if W.shape != (3 * len(bins_rgb), 2048):
    raise ValueError(f"forma inesperada para W: {W.shape}, bins={len(bins_rgb)}")
id_to_row = {mid: i for i, mid in enumerate(test_ids)}
sums = np.zeros((len(test_ids), 2048), dtype=np.float32)
counts = np.zeros(len(test_ids), dtype=np.int32)
BATCH_ESPECTROS = 8192
n_drop = 0
n_usable = 0
for batch_i, batch in enumerate(parquet.iter_batches(
    batch_size=BATCH_ESPECTROS,
    columns=["molecule_id", "precursor_mz", "ms2_mzs",
             "ms2_normalized_intensities"],
)):
    fields = batch.to_pydict()
    rows, cols, data, batch_ids = [], [], [], []
    for mid, pmz, mzs, intensidades in zip(
        fields["molecule_id"], fields["precursor_mz"], fields["ms2_mzs"],
        fields["ms2_normalized_intensities"],
    ):
        if mid is None or pmz is None or mzs is None or intensidades is None:
            n_drop += 1
            continue
        b = binear_fila(mzs, intensidades)
        if b is None or not PMZ_MIN <= float(pmz) <= PMZ_MAX:
            n_drop += 1
            continue
        bins, vals = b
        row = len(batch_ids)
        rows.extend([row] * len(bins))
        cols.extend(bins.tolist())
        data.extend(vals.tolist())
        batch_ids.append(mid)
    if batch_ids:
        Q = sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32),
             (np.asarray(rows, dtype=np.int32),
              np.asarray(cols, dtype=np.int32))),
            shape=(len(batch_ids), NBINS),
        )
        Q.sum_duplicates()
        Qf = transformar_rgb(Q, bins_rgb)
        del Q
        P_batch = np.asarray(Qf @ W, dtype=np.float32)
        del Qf
        target_rows = np.fromiter(
            (id_to_row[mid] for mid in batch_ids),
            dtype=np.int32, count=len(batch_ids),
        )
        np.add.at(sums, target_rows, P_batch)
        np.add.at(counts, target_rows, 1)
        n_usable += len(batch_ids)
        del P_batch, target_rows
    if (batch_i + 1) % 20 == 0:
        log(f"RGB: lote {batch_i + 1:,}; espectros utilizables "
            f"{n_usable:,}; descartados {n_drop:,}")

bits_by_id = {}
for i in np.flatnonzero(counts):
    scores = (sums[i] / counts[i]).reshape(1, -1)
    bits_by_id[test_ids[i]] = binarizar_predicciones_rgb(scores, top_k=100)[0]
del sums, counts, W
log(f"huellas RGB por molécula: {len(bits_by_id):,}; top-100; "
    f"espectros utilizables {n_usable:,}; descartados {n_drop:,}")

sample_path = COMP / "sample_submission.csv"
if sample_path.is_file():
    output_ids = pd.read_csv(sample_path, usecols=["molecule_id"])[
        "molecule_id"
    ].tolist()
    if set(output_ids) != set(test_ids):
        raise ValueError("IDs de sample_submission.csv no coinciden con test.parquet")
else:
    output_ids = sorted(test_ids, key=str)
if len(set(output_ids)) != len(output_ids):
    raise ValueError("IDs duplicados en la lista de salida")
log(f"filas de salida: {len(output_ids):,}")
"""),
    ("code", """# Genera beam-25 desde la huella RGB top-100. Para GPU de 4 GB se
# conserva lote 1; si hay más memoria puede aumentarse tras una prueba.
BEAM_WIDTH = 25
BATCH_MOLECULAS = 1
if BATCH_MOLECULAS < 1:
    raise ValueError("BATCH_MOLECULAS debe ser >=1")

tokenizer = Tokenizador(checkpoint["vocab"])
model = construir_modelo(
    len(checkpoint["vocab"]),
    d_model=config["d_model"],
    nhead=config["nhead"],
    num_capas=config["num_capas"],
    dim_ff=config["dim_ff"],
    dropout=config["dropout"],
    k_memoria=config["k_memoria"],
    max_len=config["max_len"],
)
model.load_state_dict(checkpoint["state_dict"])
model.to(DEVICE)
model.eval()
log(f"modelo Transformer cargado; beam={BEAM_WIDTH}; "
    f"lote={BATCH_MOLECULAS}; max_len={config['max_len']}")

def normalizar_candidatos(cands):
    cands = [s if s else "CCO" for s in cands[:BEAM_WIDTH]]
    if not cands:
        cands = ["CCO"]
    return cands + [cands[0]] * (BEAM_WIDTH - len(cands))

salida = WORK / "submission.csv"
n_sin_espectro = 0
with salida.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["molecule_id", "smiles"])
    for ini in range(0, len(output_ids), BATCH_MOLECULAS):
        ids_lote = output_ids[ini:ini + BATCH_MOLECULAS]
        ids_modelo, huellas = [], []
        for mid in ids_lote:
            bits = bits_by_id.get(mid)
            if bits is None:
                n_sin_espectro += 1
                continue
            ids_modelo.append(mid)
            huellas.append(vector_denso(bits))

        candidatos_por_id = {}
        if huellas:
            huella_t = torch.tensor(np.stack(huellas), dtype=torch.float32,
                                    device=DEVICE)
            listas, _ = generar_beam(
                model, tokenizer, huella_t, DEVICE, config["max_len"],
                beam=BEAM_WIDTH,
            )
            candidatos_por_id = {
                mid: normalizar_candidatos(cands)
                for mid, cands in zip(ids_modelo, listas)
            }

        for mid in ids_lote:
            cands = candidatos_por_id.get(mid, ["CCO"] * BEAM_WIDTH)
            writer.writerow([mid, ";".join(cands)])
        if (ini + len(ids_lote)) % 25 == 0 or ini + len(ids_lote) == len(output_ids):
            log(f"beam search: {min(ini + len(ids_lote), len(output_ids)):,}/"
                f"{len(output_ids):,}")

log(f"submission escrita: {salida}; sin espectro usable: {n_sin_espectro}")
"""),
    ("code", """# Validación estricta del contrato de submission.
submission = pd.read_csv(salida, keep_default_na=False)
n_por_fila = submission["smiles"].map(lambda value: len(value.split(";")))
checks = {
    "columnas": list(submission.columns) == ["molecule_id", "smiles"],
    "filas": len(submission) == len(test_ids),
    "ids_unicos": bool(submission["molecule_id"].is_unique),
    "ids_test": set(map(str, submission["molecule_id"])) == set(map(str, test_ids)),
    "sin_nulos_ni_vacios": bool(
        submission["smiles"].map(lambda value: bool(value.strip())).all()
    ),
    "25_candidatos_por_fila": bool((n_por_fila == BEAM_WIDTH).all()),
}
for nombre, ok in checks.items():
    print(("OK   " if ok else "FAIL ") + nombre)
md5 = hashlib.md5(salida.read_bytes()).hexdigest()
print(f"filas={len(submission):,}; candidatos={int(n_por_fila.sum()):,}; "
      f"md5={md5}; tiempo={time.time() - T0:.1f}s")
assert all(checks.values()), "submission inválida"
print("SUBMISSION VALIDADA")
"""),
    ("markdown", """## Envío

La comprobación final exige una fila por `molecule_id` de test y exactamente
25 candidatos separados por `;`. El archivo queda en
`/kaggle/working/submission.csv`. Adjunta el checkpoint OOF como input
Kaggle antes de ejecutar; el notebook no descarga modelos ni instala paquetes.

El checkpoint usado aquí produjo `MRR_novel` 0.001504 en GEN-VAL, con IC95%
pareado de la diferencia que incluye cero. El submit es exploratorio: no hay
evidencia de que supere la solución RGB actual en el test oculto.
"""),
]


def main():
    cells = []
    for i, (kind, source) in enumerate(CELDAS):
        cell = {
            "cell_type": kind,
            "id": f"b_rgb_{i:02d}",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = CODIGO / "casmi_runtime_notebook_b_rgb.ipynb"
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"notebook escrito: {output} ({output.stat().st_size:,} bytes, "
          f"{len(cells)} celdas)")


if __name__ == "__main__":
    main()
