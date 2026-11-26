"""CASMI paso 3 — 12_masas.py
Filtro por masa de precursor para el retrieval de huellas (Morgan).

Idea: el retrieval actual ordena TODO el pool (275,810 moléculas) por
Tanimoto. Pero la consulta trae su precursor_mz, y la librería completa
(full_lib) trae el precursor de cada espectro. Los candidatos con masa
compatible son los que tienen algún espectro en full_lib con precursor
dentro de ±10 ppm (piso 0.002 Da, igual que el baseline 1). Eso recorta
el pool a decenas/cientos de moléculas y deja fuera a casi todas las
que no pueden ser la verdadera por masa. Maneja aductos y dímeros solo:
el mismo aducto da el mismo precursor, sin necesidad de calcular masas
neutras ni de conocer el aducto (val_qmol_meta no lo guarda).

Salidas:
- intermedios/precursores_full_idx.npz: precursores ordenados + códigos
  de molécula (para searchsorted).
- intermedios/query_prec_<tag>.npz: por molécula de consulta, sus
  precursores distintos (valores + indptr + códigos).
- verificacion/masas_stats.json: tamaños de candidatos y supervivencia
  de la verdadera en val mol-split y spec-split.
Uso: python3 12_masas.py  (usa el venv py313 con numpy/scipy/pandas)
"""
import time, resource, json
from pathlib import Path
import numpy as np

T0 = time.time()
def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time()-T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)

BASE = Path("/home/hatch/workspace/enveda-casmi-2026")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"


def precursores_por_mol(mol, prec):
    """Agrupa precursores distintos por código de molécula.
    Devuelve (uniq_codes, values float64, indptr int64)."""
    order = np.lexsort((prec, mol))
    ms, ps = mol[order], prec[order]
    uniq, first = np.unique(ms, return_index=True)
    bounds = list(first) + [len(ms)]
    vals, indptr = [], [0]
    for i in range(len(uniq)):
        u = np.unique(ps[bounds[i]:bounds[i + 1]])
        vals.append(u)
        indptr.append(indptr[-1] + len(u))
    return (uniq.astype(np.int32),
            np.concatenate(vals).astype(np.float64),
            np.asarray(indptr, dtype=np.int64))


def candidatos(prec_sorted, mol_sorted, queries_prec, tol_ppm=10.0, tol_min=0.002):
    """Para cada precursor de consulta: códigos únicos con algún espectro
    en [q-tol, q+tol]. Devuelve lista de arrays (uno por precursor)."""
    out = []
    for q in queries_prec:
        tol = max(float(q) * tol_ppm * 1e-6, tol_min)
        lo = int(np.searchsorted(prec_sorted, q - tol))
        hi = int(np.searchsorted(prec_sorted, q + tol))
        out.append(np.unique(mol_sorted[lo:hi]) if hi > lo else np.empty(0, dtype=np.int32))
    return out


def main():
    fl = np.load(INTER / "full_lib_meta.npz", allow_pickle=True)
    fl_mol = fl["mol"].astype(np.int32)
    fl_prec = fl["precursor_mz"].astype(np.float64)
    order = np.argsort(fl_prec, kind="stable")
    prec_sorted = fl_prec[order]
    mol_sorted = fl_mol[order]
    del fl, fl_mol, fl_prec, order
    np.savez(INTER / "precursores_full_idx.npz",
             prec=prec_sorted, mol=mol_sorted)
    log(f"índice precursores full_lib: {len(prec_sorted):,} espectros, "
        f"prec {prec_sorted.min():.1f}-{prec_sorted.max():.1f}")

    stats = {}
    # --- consultas de validación: mol-split y spec-split ---
    for tag, meta in (("val_mol", "val_qmol_meta"), ("val_spec", "val_qspec_meta")):
        m = np.load(INTER / f"{meta}.npz", allow_pickle=True)
        mq = m["mol"].astype(np.int32)
        pq = m["precursor_mz"].astype(np.float64)
        codes, vals, indptr = precursores_por_mol(mq, pq)
        np.savez(INTER / f"query_prec_{tag}.npz",
                 mol=codes, prec=vals, indptr=indptr)
        log(f"query_prec_{tag}: {len(codes):,} moléculas, "
            f"precursores distintos/mol media {np.diff(indptr).mean():.2f}")
        # candidatos por molécula (unión de ventanas de sus precursores)
        for tol_ppm, tol_min, tname in ((10.0, 0.002, "ppm10"), (20.0, 0.005, "ppm20")):
            cand_n, sobrevive = [], []
            for i in range(len(codes)):
                cands = set()
                for q in vals[indptr[i]:indptr[i + 1]]:
                    for c in candidatos(prec_sorted, mol_sorted, [q], tol_ppm, tol_min)[0]:
                        cands.add(int(c))
                cand_n.append(len(cands))
                sobrevive.append(int(codes[i]) in cands)
            cand_n = np.asarray(cand_n)
            sobrevive = np.asarray(sobrevive)
            log(f"  {tag} {tname}: candidatos/mol mediana {np.median(cand_n):.0f} "
                f"media {cand_n.mean():.0f} p90 {np.percentile(cand_n, 90):.0f} "
                f"máx {cand_n.max()}, verdad sobrevive {sobrevive.mean():.4f} "
                f"({int(sobrevive.sum())}/{len(sobrevive)})")
            stats[f"{tag}_{tname}"] = {
                "cand_mediana": float(np.median(cand_n)),
                "cand_media": float(cand_n.mean()),
                "cand_p90": float(np.percentile(cand_n, 90)),
                "cand_max": int(cand_n.max()),
                "verdad_sobrevive": float(sobrevive.mean()),
                "n": int(len(codes)),
            }
        del m, mq, pq

    # --- consultas de test: por molecule_id ---
    mm = np.load(INTER / "test_q_paso2_meta.npz", allow_pickle=True)
    q_id = mm["molecule_id"]
    q_pmz = mm["precursor_mz"].astype(np.float64)
    order = np.lexsort((q_pmz, q_id))
    qs, ps = q_id[order], q_pmz[order]
    uniq, first = np.unique(qs, return_index=True)
    bounds = list(first) + [len(qs)]
    vals, indptr = [], [0]
    for i in range(len(uniq)):
        u = np.unique(ps[bounds[i]:bounds[i + 1]])
        vals.append(u)
        indptr.append(indptr[-1] + len(u))
    np.savez(INTER / "query_prec_test.npz",
             molecule_id=uniq, prec=np.concatenate(vals).astype(np.float64),
             indptr=np.asarray(indptr, dtype=np.int64))
    log(f"query_prec_test: {len(uniq)} moléculas, "
        f"precursores distintos/mol media {np.diff(indptr).mean():.2f}")
    cand_n = []
    for i in range(len(uniq)):
        cands = set()
        for q in np.concatenate(vals)[indptr[i]:indptr[i + 1]]:
            for c in candidatos(prec_sorted, mol_sorted, [q], 10.0, 0.002)[0]:
                cands.add(int(c))
        cand_n.append(len(cands))
    cand_n = np.asarray(cand_n)
    log(f"  test ppm10: candidatos/mol mediana {np.median(cand_n):.0f} "
        f"media {cand_n.mean():.0f} p90 {np.percentile(cand_n, 90):.0f} máx {cand_n.max()}")
    stats["test_ppm10"] = {
        "cand_mediana": float(np.median(cand_n)), "cand_media": float(cand_n.mean()),
        "cand_p90": float(np.percentile(cand_n, 90)), "cand_max": int(cand_n.max()),
        "n": int(len(uniq)),
    }
    (VERIF / "masas_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"FIN 12_masas: {stats}")

if __name__ == "__main__":
    main()
