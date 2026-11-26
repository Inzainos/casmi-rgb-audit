"""CASMI paso 2 — 04_huellas.py
Calcula la huella de Morgan (2048 bits, radio 2) de cada molécula de train
a partir de su SMILES representativo (mapa_smiles_full.parquet, el más
completo: 275,303 moléculas; las 507 sin ningún espectro válido no tienen
SMILES y no pueden ser candidatas).

Salida:
- intermedios/huellas_pool.npz: CSR bool (275,810 x 2048); la fila i es el
  código de molécula i (posición en mol_index.npy). Filas vacías = molécula
  sin huella (SMILES no parseable o sin SMILES).
- intermedios/huellas_freq.npz: frecuencias de bit (2048,) float32
  (para el baseline nulo: huella media).
- verificacion/huellas_stats.json: estadísticas y fallos de parseo.
"""
import time, resource, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

T0 = time.time()
def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time()-T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)

BASE = Path("/home/hatch/workspace/snt-taxonomia/kaggle-casmi")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"
NBITS = 2048
RADIO = 2

def main():
    mols = np.load(VERIF / "mol_index.npy", allow_pickle=True)
    n_mols = len(mols)
    code = {m: i for i, m in enumerate(mols)}
    mapa = pd.read_parquet(INTER / "mapa_smiles_full.parquet")
    log(f"mapa: {len(mapa):,} SMILES representativos de {n_mols:,} moléculas")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIO, fpSize=NBITS)

    rows, cols = [], []
    n_fallo = 0
    fallos = []
    t1 = time.time()
    for k, (ik, sm) in enumerate(zip(mapa["inchikey14"], mapa["smiles_rep"])):
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            n_fallo += 1
            if len(fallos) < 20:
                fallos.append((ik, sm))
            continue
        fp = gen.GetFingerprint(mol)
        bits = np.fromiter(fp.GetOnBits(), dtype=np.int32, count=-1)
        c = code[ik]
        rows.append(np.full(len(bits), c, dtype=np.int32))
        cols.append(bits)
        if (k + 1) % 25_000 == 0:
            el = time.time() - t1
            log(f"huellas: {k+1:,}/{len(mapa):,} ({el:.0f}s, "
                f"~{el/(k+1)*len(mapa):.0f}s total estimado)")
    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int32)
    cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int32)
    data = np.ones(len(rows), dtype=bool)
    X = sparse.csr_matrix((data, (rows, cols)), shape=(n_mols, NBITS))
    X.sort_indices()
    log(f"CSR huellas: {X.shape[0]:,} x {X.shape[1]:,}, nnz {X.nnz:,}")
    sparse.save_npz(INTER / "huellas_pool.npz", X)

    # frecuencias de bit (baseline nulo) y estadísticas
    n_bits = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    freq = (n_bits / max(X.shape[0], 1)).astype(np.float32)
    np.savez(INTER / "huellas_freq.npz", freq=freq, n_bits=n_bits.astype(np.int64))
    bits_por_mol = np.diff(X.indptr).astype(np.int64)
    con_fp = bits_por_mol > 0
    stats = {
        "n_moleculas": int(n_mols),
        "n_con_smiles": int(len(mapa)),
        "n_con_huella": int(con_fp.sum()),
        "n_sin_huella": int((~con_fp).sum()),
        "n_fallo_parseo": int(n_fallo),
        "ejemplos_fallo": fallos,
        "nnz_total": int(X.nnz),
        "bits_por_mol_media": float(bits_por_mol[con_fp].mean()),
        "bits_por_mol_mediana": float(np.median(bits_por_mol[con_fp])),
        "bits_por_mol_p10": float(np.percentile(bits_por_mol[con_fp], 10)),
        "bits_por_mol_p90": float(np.percentile(bits_por_mol[con_fp], 90)),
        "bits_por_mol_min": int(bits_por_mol[con_fp].min()),
        "bits_por_mol_max": int(bits_por_mol[con_fp].max()),
        "bits_con_uso_0": int((n_bits == 0).sum()),
        "bits_con_uso_lt_10": int((n_bits < 10).sum()),
        "bits_con_uso_lt_100": int((n_bits < 100).sum()),
        "bits_con_uso_ge_1pct": int((n_bits >= 0.01 * n_mols).sum()),
        "freq_media_bits_con_uso": float(freq[n_bits > 0].mean()),
    }
    (VERIF / "huellas_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"FIN: {stats['n_con_huella']:,} con huella, {stats['n_sin_huella']:,} sin huella, "
        f"{n_fallo} fallos de parseo; bits/mol media {stats['bits_por_mol_media']:.1f}, "
        f"bits sin uso {stats['bits_con_uso_0']}")

if __name__ == "__main__":
    main()
