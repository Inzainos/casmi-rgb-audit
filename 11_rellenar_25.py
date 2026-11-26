"""CASMI baseline 1 — 11_rellenar_25.py
Rellena a 25 candidatos exactos las 33 filas cortas de salida/submission.csv.

- Misma búsqueda que 03_predecir_test.py: mismo bineado (top-100 picos,
  bins 0.01 Da), misma ventana de precursor (±10 ppm, piso 0.002 Da),
  mismo coseno (raíz cuadrada + L2, estándar GNPS), misma agregación por
  molécula (máxima similitud por inchikey14, rank_molecule).
- Diferencia única: NO se corta la lista rankeada en 25; se toman los
  candidatos rankeados DESPUÉS de los ya listados, en orden de similitud,
  y se añaden al final de la fila hasta llegar a 25. Los candidatos
  originales NO se tocan ni se reordenan.
- Si la lista completa dedupada de una molécula tiene <25 entradas (no hay
  más candidatos reales en la ventana): se completa repitiendo su candidato
  top-1 (mayor similitud). NO se usa "CCO" y NO se ensancha la ventana.
- Salida: archivo NUEVO salida/submission_relleno25.csv. NUNCA se toca
  salida/submission.csv (se verifica su md5 antes y después).
- NO envía nada a Kaggle.
"""
import hashlib, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
sys.path.insert(0, str(Path(__file__).parent))
from casmi_lib import log, load_library, row_stats, score_query, rank_molecule
from importlib import import_module
_construir = import_module("01_construir")

BASE = Path("/home/hatch/workspace/snt-taxonomia/kaggle-casmi")
DATOS = BASE / "datos"
INTER = BASE / "intermedios"
SALIDA = BASE / "salida"
VERIF = BASE / "verificacion"
MD5_ORIGINAL = "7a7da389417fb162039a94394d76e1ee"


def md5_de(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main():
    md5_antes = md5_de(SALIDA / "submission.csv")
    assert md5_antes == MD5_ORIGINAL, f"submission.csv cambió: {md5_antes}"
    log(f"submission.csv original verificado: md5 {md5_antes}")

    original = pd.read_csv(SALIDA / "submission.csv")
    assert len(original) == 400, len(original)
    orig = {r.molecule_id: r.smiles.split(";")
            for r in original.itertuples()}
    cortas = [mid for mid, s in orig.items() if len(s) < 25]
    assert len(cortas) == 33, f"cortas: {len(cortas)} (se esperaban 33)"
    log(f"filas cortas a rellenar: {len(cortas)}")

    X, prec, mol_sorted, _ = load_library(INTER, "full_lib")
    rowsum, S_row, norm_cos = row_stats(X)

    mapa = pd.read_parquet(INTER / "mapa_smiles_full.parquet")
    mol_names = np.load(VERIF / "mol_index.npy", allow_pickle=True)
    rep = dict(zip(mapa["inchikey14"], mapa["smiles_rep"]))
    log(f"mapa SMILES: {len(rep):,} moléculas con representativo")

    # ---- binear test (idéntico a 03) ----
    test = pd.read_parquet(DATOS / "test.parquet")
    rows, cols, data = [], [], []
    q_mol_id, q_pmz = [], []
    n_drop = 0
    for i, r in test.iterrows():
        pmz = float(r["precursor_mz"])
        b = _construir.binear_fila(r["ms2_mzs"], r["ms2_normalized_intensities"])
        if b is None or not (100.0 <= pmz <= 2000.0):
            n_drop += 1
            continue
        bins, vals = b
        k = len(q_mol_id)
        rows.extend([k] * len(bins))
        cols.extend(bins.tolist())
        data.extend(vals.tolist())
        q_mol_id.append(r["molecule_id"])
        q_pmz.append(pmz)
    Q = sparse.csr_matrix((np.asarray(data, dtype=np.float32),
                           (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
                          shape=(len(q_mol_id), 200_000))
    Q.sum_duplicates()
    q_mol_id = np.asarray(q_mol_id, dtype=object)
    q_pmz = np.asarray(q_pmz, dtype=np.float64)
    log(f"test bineado: {Q.shape[0]:,} espectros, descartados {n_drop}")

    order = np.argsort(q_mol_id, kind="stable")
    q_mol_s = q_mol_id[order]
    uniq, starts = np.unique(q_mol_s, return_index=True)
    bounds = list(starts) + [len(q_mol_s)]
    pos = {mid: ui for ui, mid in enumerate(uniq)}

    # ---- rellenar solo las 33 cortas ----
    detalle = []
    nuevas = {}
    t0 = time.time()
    for n, mid in enumerate(cortas, 1):
        ui = pos[mid]
        s, e = bounds[ui], bounds[ui + 1]
        best, mols = [], []
        for j in order[s:e]:
            m, c, _en = score_query(X, prec, mol_sorted, rowsum, S_row, norm_cos,
                                    Q[j], float(q_pmz[j]), "ppm")
            if m is None:
                continue
            best.append(c)
            mols.append(m)
        assert best, f"sin candidatos para {mid} (03 sí los encontró)"
        ranking = rank_molecule(best, mols)  # lista COMPLETA, sin [:25]
        lista = []
        for code in ranking:
            sm = rep.get(mol_names[code])
            if sm:
                lista.append(sm)
        n_orig = len(orig[mid])
        # Seguridad: la lista completa debe empezar EXACTAMENTE con los
        # candidatos originales, en el mismo orden (misma búsqueda que 03).
        assert lista[:n_orig] == orig[mid], (
            f"{mid}: la lista completa no reproduce los originales "
            f"(primeros {n_orig})")
        siguientes = lista[n_orig:25]
        n_reales = len(siguientes)
        n_rep = 25 - n_orig - n_reales
        fila = orig[mid] + siguientes
        if n_rep > 0:
            fila = fila + [orig[mid][0]] * n_rep
        assert len(fila) == 25, (mid, len(fila))
        nuevas[mid] = fila
        detalle.append({"molecule_id": mid, "n_original": n_orig,
                        "n_lista_completa": len(lista),
                        "n_reales_anadidos": n_reales,
                        "n_repeticiones_top1": n_rep,
                        "top1_smiles": orig[mid][0]})
        if n % 10 == 0 or n == len(cortas):
            log(f"relleno: {n}/{len(cortas)} moléculas ({time.time()-t0:.0f}s)")
    log(f"relleno completo: {len(nuevas)} filas en {time.time()-t0:.0f}s")

    # ---- escribir archivo nuevo, mismo orden que sample_submission ----
    out_rows = []
    for r in original.itertuples():
        sm = nuevas.get(r.molecule_id, r.smiles.split(";"))
        out_rows.append({"molecule_id": r.molecule_id, "smiles": ";".join(sm)})
    sub = pd.DataFrame(out_rows)
    sample = pd.read_csv(DATOS / "sample_submission.csv")
    assert [r for r in sub["molecule_id"]] == [r for r in sample["molecule_id"]]
    assert len(sub) == 400
    n_sm = sub["smiles"].apply(lambda s: len(s.split(";")))
    assert (n_sm == 25).all(), n_sm.describe().to_dict()
    ruta_nueva = SALIDA / "submission_relleno25.csv"
    sub.to_csv(ruta_nueva, index=False)
    md5_nuevo = md5_de(ruta_nueva)
    assert md5_nuevo != MD5_ORIGINAL

    det = pd.DataFrame(detalle)
    det.to_csv(VERIF / "relleno25_detalle.csv", index=False)
    resumen = {
        "archivo": "salida/submission_relleno25.csv",
        "md5_nuevo": md5_nuevo,
        "md5_original": MD5_ORIGINAL,
        "filas": 400,
        "filas_rellenadas": len(cortas),
        "filas_intactas": 400 - len(cortas),
        "reales_anadidos_total": int(det["n_reales_anadidos"].sum()),
        "repeticiones_top1_total": int(det["n_repeticiones_top1"].sum()),
        "filas_solo_reales": int(((det["n_repeticiones_top1"] == 0)).sum()),
        "filas_con_repeticiones": int(((det["n_repeticiones_top1"] > 0)).sum()),
        "filas_sin_mas_candidatos_reales": int(((det["n_reales_anadidos"] == 0)).sum()),
    }
    with open(VERIF / "relleno25.json", "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)
    log(f"escrito {ruta_nueva} md5 {md5_nuevo}")
    log(f"resumen: {resumen}")
    md5_despues = md5_de(SALIDA / "submission.csv")
    assert md5_despues == MD5_ORIGINAL, "submission.csv CAMBIÓ durante la corrida"
    log("submission.csv original intacto tras la corrida")
    log("RELLENO COMPLETO (sin envío a Kaggle)")


if __name__ == "__main__":
    main()
