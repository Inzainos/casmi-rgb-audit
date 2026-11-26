"""CASMI baseline 1 — 01_construir.py
Construye la librería espectral bineada (CSR) y los conjuntos de consulta.

Diseño (memoria):
- Lee train.parquet por chunks (200k filas); nunca carga los 2.5M de espectros.
- Por espectro: top-100 picos por intensidad, bins de 0.01 Da (0-2000 Da),
  intensidades sumadas por bin. Guarda UNA CSR con intensidades crudas
  bineadas (float32); coseno y entropía se derivan de ella en 02_validar.py.
- Filtros: precursor_mz en [100, 2000] Da, num_peaks >= 3, intensidad > 0.
- Split (semilla 42): 5% de las moléculas (inchikey14) reservadas (mol-split);
  además 5% de las moléculas con >=4 espectros para el diagnóstico spec-split
  (se retienen ~3 espectros por molécula como consultas, el resto queda en librería).

Modos:
  --modo split  : solo escribe los archivos del split y el resumen.
  --modo val    : librería de validación + queries mol-split + queries spec-split.
  --modo full   : librería completa (todo train filtrado) para el envío final.
"""
import argparse, time, resource, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

T0 = time.time()
def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time()-T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)

BASE = Path("/home/hatch/workspace/snt-taxonomia/kaggle-casmi")
DATOS = BASE / "datos"
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"
for d in (INTER, VERIF):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
BIN_W = 0.01
NBINS = 200_000          # 0..2000 Da a 0.01 Da
TOPK = 100               # picos por espectro (por intensidad)
CHUNK = 100_000
PMZ_MIN, PMZ_MAX = 100.0, 2000.0
NPEAK_MIN = 3


def fase_split():
    """Lee inchikey14, define los splits y los guarda (o los carga si ya existen)."""
    cache = VERIF / "split_arrays.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        mols = z["mols"]
        n_per_mol = dict(zip(mols.tolist(), z["n_per_mol"].tolist()))
        val_set = set(z["val_mols"].tolist())
        spec_set = set(z["spec_mols"].tolist())
        mol_index = {m: i for i, m in enumerate(mols)}
        log(f"split cargado de disco: {len(mols):,} moléculas, val {len(val_set):,}, spec {len(spec_set):,}")
        return mols, mol_index, val_set, spec_set, n_per_mol
    log("fase split: leyendo inchikey14...")
    ik = pq.read_table(str(DATOS / "train.parquet"), columns=["inchikey14"]).column("inchikey14").to_pandas()
    vc = ik.value_counts()
    mols = np.sort(vc.index.to_numpy())          # 275,810 moléculas, orden determinista
    n_per_mol = vc.loc[mols].to_numpy()
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(mols))
    n_val = int(round(0.05 * len(mols)))
    val_mols = np.sort(mols[perm[:n_val]])
    # spec-split: 5% de las moléculas con >=4 espectros, EXCLUYENDO las de validación
    val_set_tmp = set(val_mols.tolist())
    cand = np.array([i for i in np.where(n_per_mol >= 4)[0] if mols[i] not in val_set_tmp])
    n_spec = int(round(0.05 * len(cand)))
    spec_mols = np.sort(mols[cand[rng.permutation(len(cand))[:n_spec]]])
    assert len(set(val_mols) & set(spec_mols)) == 0, "los splits deben ser disjuntos"
    # índice de molécula (código entero) y nº de espectros por molécula
    mol_index = {m: i for i, m in enumerate(mols)}
    np.save(VERIF / "mol_index.npy", mols)       # array ordenado de inchikey14
    np.savez(cache, mols=mols, n_per_mol=n_per_mol,
             val_mols=np.asarray(sorted(val_mols)), spec_mols=np.asarray(sorted(spec_mols)))
    pd.Series(val_mols).to_csv(VERIF / "val_molecules.txt", index=False, header=False)
    pd.Series(spec_mols).to_csv(VERIF / "spec_molecules.txt", index=False, header=False)
    resumen = {
        "n_moleculas": int(len(mols)), "n_val_moleculas": int(len(val_mols)),
        "n_spec_moleculas": int(len(spec_mols)),
        "espectros_val_moleculas": int(n_per_mol[np.isin(mols, val_mols)].sum()),
        "espectros_spec_moleculas": int(n_per_mol[np.isin(mols, spec_mols)].sum()),
        "semilla": SEED,
    }
    (VERIF / "split_resumen.json").write_text(json.dumps(resumen, indent=2))
    log(f"split: {resumen}")
    return mols, mol_index, set(val_mols.tolist()), set(spec_mols.tolist()), vc.to_dict()


def binear_fila(mzs, ints):
    """Top-100 por intensidad, bins 0.01 Da, suma por bin. Devuelve (bins, vals)."""
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


def _destino_fila(ik, pmz, npeak, g, modo, val_set, spec_set, n_per_mol, holdout_spec=None):
    """Destino de una fila (None = descartada). Idéntica lógica en la pasada escalar.
    holdout_spec: máscara booleana por índice global de fila (solo modo val):
    para moléculas spec-split marca los ~3 espectros retenidos como consultas,
    garantizando que quede >=1 espectro en la librería."""
    if not (PMZ_MIN <= pmz <= PMZ_MAX) or npeak < NPEAK_MIN:
        return None
    n_mol = n_per_mol.get(ik, 1)
    h = ((g * 2654435761) % (2 ** 32)) / (2 ** 32)
    if modo == "val":
        if ik in val_set:
            if h >= 3.0 / max(n_mol, 1) and n_mol > 3:
                return None
            return "qmol"
        if ik in spec_set:
            if holdout_spec is not None:
                return "qspec" if bool(holdout_spec[g]) else "lib"
            return "qspec" if h < 3.0 / max(n_mol, 1) else "lib"
        return "lib"
    return "lib"


def calcular_holdout_spec(mols, spec_set):
    """Máscara de holdout spec-split: por molécula, los min(3, n_kept-1) espectros
    con menor hash h (entre los que pasan los filtros) van a consultas; el resto
    queda en librería. Garantiza >=1 espectro en librería por molécula.
    Se calcula sobre los arrays acumulados (sin releer el archivo)."""
    cache = VERIF / "holdout_spec.npy"
    if cache.exists():
        m = np.load(cache)
        log(f"holdout spec cargado de disco: {int(m.sum()):,} filas marcadas")
        return m
    log("calculando holdout spec-split...")
    codes_all, h_all, keep_all = [], [], []
    pf = pq.ParquetFile(str(DATOS / "train.parquet"))
    offset = 0
    for batch in pf.iter_batches(batch_size=250_000,
                                 columns=["inchikey14", "precursor_mz", "num_peaks"]):
        df = batch.to_pandas()
        codes = pd.Categorical(df["inchikey14"], categories=mols).codes.astype(np.int32)
        pmz = df["precursor_mz"].to_numpy(dtype=np.float64)
        npeak = df["num_peaks"].to_numpy(dtype=np.int64)
        keep = (pmz >= PMZ_MIN) & (pmz <= PMZ_MAX) & (npeak >= NPEAK_MIN) & (codes >= 0)
        g = np.arange(offset, offset + len(df), dtype=np.int64)
        h = ((g * 2654435761) % (2 ** 32)) / (2 ** 32)
        codes_all.append(codes)
        h_all.append(h)
        keep_all.append(keep)
        offset += len(df)
        del df
    codes = np.concatenate(codes_all)
    h = np.concatenate(h_all)
    keep = np.concatenate(keep_all)
    del codes_all, h_all, keep_all
    spec_codes = np.asarray([np.searchsorted(mols, m) for m in spec_set], dtype=np.int32)
    is_spec_code = np.zeros(len(mols), dtype=bool)
    is_spec_code[spec_codes] = True
    order = np.lexsort((h, codes))          # ordena por código y luego por h
    cs, hs, ks = codes[order], h[order], keep[order]
    mask_sorted = np.zeros(len(cs), dtype=bool)
    uniq, starts = np.unique(cs, return_index=True)
    bounds = list(starts) + [len(cs)]
    for ui, code in enumerate(uniq):
        if not is_spec_code[code]:
            continue
        s, e = bounds[ui], bounds[ui + 1]
        kept_pos = np.where(ks[s:e])[0]
        n_hold = min(3, max(0, len(kept_pos) - 1))
        if n_hold > 0:
            mask_sorted[s + kept_pos[:n_hold]] = True
    mask = np.zeros(len(cs), dtype=bool)
    mask[order] = mask_sorted
    np.save(cache, mask)
    log(f"holdout spec calculado: {int(mask.sum()):,} filas marcadas como consultas")
    return mask


def pasada_escalar(modo, mols, val_set, spec_set, n_per_mol):
    """Pasa escalar vectorizada: cotas superiores de filas y nnz por destino.
    nnz_fila <= min(num_peaks, TOPK) (el bineado solo puede fusionar picos).
    Devuelve (rows_upper, nnz_upper, holdout_spec)."""
    holdout = calcular_holdout_spec(mols, spec_set) if modo == "val" else None
    n_per_mol_arr = np.asarray([n_per_mol[m] for m in mols], dtype=np.int64)
    is_val_code = np.zeros(len(mols), dtype=bool)
    is_spec_code = np.zeros(len(mols), dtype=bool)
    for m in val_set:
        is_val_code[np.searchsorted(mols, m)] = True
    for m in spec_set:
        is_spec_code[np.searchsorted(mols, m)] = True
    rows_upper = {"lib": 0, "qmol": 0, "qspec": 0}
    nnz_upper = {"lib": 0, "qmol": 0, "qspec": 0}
    pf = pq.ParquetFile(str(DATOS / "train.parquet"))
    offset = 0
    for batch in pf.iter_batches(batch_size=250_000,
                                 columns=["inchikey14", "precursor_mz", "num_peaks"]):
        df = batch.to_pandas()
        codes = pd.Categorical(df["inchikey14"], categories=mols).codes
        pmz = df["precursor_mz"].to_numpy(dtype=np.float64)
        npeak = df["num_peaks"].to_numpy(dtype=np.int64)
        keep = (pmz >= PMZ_MIN) & (pmz <= PMZ_MAX) & (npeak >= NPEAK_MIN) & (codes >= 0)
        c = np.clip(codes, 0, None)
        is_val = is_val_code[c]
        is_spec = is_spec_code[c]
        n_mol = n_per_mol_arr[c]
        g = np.arange(offset, offset + len(df), dtype=np.int64)
        h = ((g * 2654435761) % (2 ** 32)) / (2 ** 32)
        thr = 3.0 / np.maximum(n_mol, 1)
        if modo == "val":
            m_qmol = keep & is_val & ((h < thr) | (n_mol <= 3))
            m_qspec = keep & is_spec & holdout[offset:offset + len(df)]
            m_lib = keep & (~is_val) & (~m_qspec)
        else:
            m_lib = keep
            m_qmol = np.zeros(len(df), dtype=bool)
            m_qspec = np.zeros(len(df), dtype=bool)
        capped = np.minimum(npeak, TOPK)
        for m, name in ((m_lib, "lib"), (m_qmol, "qmol"), (m_qspec, "qspec")):
            rows_upper[name] += int(m.sum())
            nnz_upper[name] += int(capped[m].sum())
        offset += len(df)
        del df
    log(f"pasada escalar modo={modo}: filas {rows_upper} | nnz cota {nnz_upper}")
    return rows_upper, nnz_upper, holdout


def construir(modo, mols, mol_index, val_set, spec_set, n_per_mol):
    """Pasa train en chunks y construye las CSR pedidas (arrays prealocados)."""
    log(f"construir modo={modo}")
    rows_upper, nnz_upper, holdout = pasada_escalar(modo, mols, val_set, spec_set, n_per_mol)
    dests = ["lib"] + (["qmol", "qspec"] if modo == "val" else [])
    # prealocación (numpy, no listas de Python: 137M nnz en listas = OOM)
    R = {d: np.empty(nnz_upper[d], dtype=np.int32) for d in dests}
    C = {d: np.empty(nnz_upper[d], dtype=np.int32) for d in dests}
    D = {d: np.empty(nnz_upper[d], dtype=np.float32) for d in dests}
    prec = {d: np.empty(rows_upper[d], dtype=np.float32) for d in dests}
    mol = {d: np.empty(rows_upper[d], dtype=np.int32) for d in dests}
    ing = {d: np.empty(rows_upper[d], dtype=np.int8) for d in dests}
    off = {d: 0 for d in dests}
    rowidx = {d: 0 for d in dests}
    ingest_cat, ingest_code = [], {}
    smiles_count = {}
    stats = {"filas_leidas": 0, "filtradas_pmz": 0, "filtradas_npeak": 0, "sin_picos": 0,
             "a_libreria": 0, "a_qmol": 0, "a_qspec": 0}
    pf = pq.ParquetFile(str(DATOS / "train.parquet"))
    row_global = 0
    n_chunks = 0
    for batch in pf.iter_batches(batch_size=CHUNK,
                                 columns=["inchikey14", "normalized_smiles", "precursor_mz",
                                          "ms2_mzs", "ms2_normalized_intensities", "num_peaks", "ingest_lib"]):
        df = batch.to_pandas()
        n_chunks += 1
        for pos in range(len(df)):
            ik = df["inchikey14"].iloc[pos]
            stats["filas_leidas"] += 1
            pmz = float(df["precursor_mz"].iloc[pos])
            npeak = int(df["num_peaks"].iloc[pos])
            dest = _destino_fila(ik, pmz, npeak, row_global + pos, modo, val_set, spec_set, n_per_mol, holdout)
            if dest is None:
                if not (PMZ_MIN <= pmz <= PMZ_MAX):
                    stats["filtradas_pmz"] += 1
                elif npeak < NPEAK_MIN:
                    stats["filtradas_npeak"] += 1
                continue
            sm = df["normalized_smiles"].iloc[pos]
            d = smiles_count.setdefault(ik, {})
            d[sm] = d.get(sm, 0) + 1
            r = binear_fila(df["ms2_mzs"].iloc[pos], df["ms2_normalized_intensities"].iloc[pos])
            if r is None:
                stats["sin_picos"] += 1
                continue
            bins, vals = r
            n = len(bins)
            o = off[dest]
            k = rowidx[dest]
            R[dest][o:o + n] = k
            C[dest][o:o + n] = bins
            D[dest][o:o + n] = vals
            off[dest] += n
            prec[dest][k] = pmz
            mol[dest][k] = mol_index[ik]
            s_ing = df["ingest_lib"].iloc[pos]
            code = ingest_code.get(s_ing)
            if code is None:
                code = len(ingest_cat)
                ingest_cat.append(s_ing)
                ingest_code[s_ing] = code
            ing[dest][k] = code
            rowidx[dest] += 1
            stats["a_libreria" if dest == "lib" else ("a_qmol" if dest == "qmol" else "a_qspec")] += 1
        row_global += len(df)
        del df
        log(f"chunk {n_chunks}: leídas {stats['filas_leidas']:,} | lib {stats['a_libreria']:,} | "
            f"qmol {stats['a_qmol']:,} | qspec {stats['a_qspec']:,} | filt pmz {stats['filtradas_pmz']:,} "
            f"npeak {stats['filtradas_npeak']:,} sin_picos {stats['sin_picos']:,}")
    # guardar cada salida como CSR
    for nombre in dests:
        o = off[nombre]
        n_row = rowidx[nombre]
        csr = sparse.csr_matrix(
            (D[nombre][:o], (R[nombre][:o], C[nombre][:o])),
            shape=(n_row, NBINS))
        csr.sum_duplicates()
        tag = f"{modo}_{nombre}"
        sparse.save_npz(INTER / f"{tag}.npz", csr)
        np.savez(INTER / f"{tag}_meta.npz",
                 precursor_mz=prec[nombre][:n_row],
                 mol=mol[nombre][:n_row],
                 ingest_lib=np.asarray([ingest_cat[c] for c in ing[nombre][:n_row]], dtype=object))
        log(f"guardado {tag}: {n_row:,} filas, {csr.nnz:,} nnz (cota {nnz_upper[nombre]:,}), "
            f"{csr.data.nbytes/1e6:.0f} MB datos")
    # mapa SMILES representativo
    rep = {ik: max(c.items(), key=lambda kv: kv[1])[0] for ik, c in smiles_count.items()}
    pd.DataFrame({"inchikey14": list(rep.keys()), "smiles_rep": list(rep.values())}).to_parquet(
        INTER / f"mapa_smiles_{modo}.parquet", index=False)
    (VERIF / f"build_stats_{modo}.json").write_text(json.dumps(stats, indent=2))
    log(f"FIN modo={modo}: {stats}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["split", "val", "full"], required=True)
    args = ap.parse_args()
    mols, mol_index, val_set, spec_set, n_per_mol = fase_split()
    if args.modo != "split":
        construir(args.modo, mols, mol_index, val_set, spec_set, n_per_mol)
    log("OK")
