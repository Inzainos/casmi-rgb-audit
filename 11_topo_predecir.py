"""CASMI paso 2 — 11_topo_predecir.py
Predice huellas con el modelo topológico para val (mol-split) y spec
(spec-split) y agrega por molécula (media de los scores de sus espectros,
igual que los otros modelos del paso 2).

Salidas (intermedios/):
- P_val_topo.npy (13,748 x 2048), P_val_mol_topo.npy (códigos de molécula)
- P_spec_topo.npy (9,936 x 2048), P_spec_mol_topo.npy
- diag_val_topo.npz, diag_spec_topo.npz (conteos por espectro: presentes
  Nivel 1 / Nivel 2, capturas, nº de picos)
- verificacion/topo_predecir.json (tiempos, memoria, medias)

Uso: python3 11_topo_predecir.py [--scoring binomial|gaussiano] [--sufijo _v2]
El sufijo distingue versiones en los archivos de salida (la v1 es gaussiana).
"""
import time, resource, json
from pathlib import Path
import numpy as np
from scipy import sparse
import sys
sys.path.insert(0, str(Path(__file__).parent))
from casmi_lib import log, load_queries
from topo_lib import cargar_baseline, predecir_huellas, agregar_max_por_mol
from importlib import import_module

T0 = time.time()
_m06 = import_module("06_entrenar_lineal")
BASE = Path("/home/hatch/workspace/snt-taxonomia/kaggle-casmi")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", choices=["binomial", "gaussiano"], default="binomial")
    ap.add_argument("--sufijo", default="")
    args = ap.parse_args()
    baseline = cargar_baseline(INTER)
    log(f"baseline topológico: M={baseline['M']}, level1={int(baseline['level1'].sum())} bits, "
        f"d mediana={float(np.median(baseline['cohens_d'])):.3f}, "
        f"normalización={baseline.get('normalizacion')}, scoring={args.scoring}")
    bins_sel = np.load(INTER / "bins_sel_paso2.npy")
    out = {}
    for tag, npz in (("val", "val_qmol"), ("spec", "val_qspec")):
        Q, _prec, mol_q, _ing = load_queries(INTER, npz)
        P, diag = predecir_huellas(Q, bins_sel, baseline, scoring=args.scoring)
        del Q
        Pm, uniq, counts = _m06.agregar_por_mol(P, mol_q.astype(np.int32))
        Pm_max, uniq_max, _c = agregar_max_por_mol(P, mol_q.astype(np.int32))
        assert (uniq_max == uniq).all()
        np.save(INTER / f"P_{tag}_topo{args.sufijo}.npy", Pm)
        np.save(INTER / f"P_{tag}_topo_max{args.sufijo}.npy", Pm_max)
        np.save(INTER / f"P_{tag}_mol_topo{args.sufijo}.npy", uniq)
        np.savez(INTER / f"diag_{tag}_topo{args.sufijo}.npz", **diag)
        pres_mol = (Pm > 0.5).sum(axis=1)
        pres_mol_max = (Pm_max > 0.5).sum(axis=1)
        out[tag] = {
            "n_moleculas": int(Pm.shape[0]),
            "pred_media": float(Pm.mean()),
            "pred_max": float(Pm.max()),
            "bits_presentes_por_mol_media": float(pres_mol.mean()),
            "bits_presentes_por_mol_mediana": float(np.median(pres_mol)),
            "bits_presentes_por_mol_max_media": float(pres_mol_max.mean()),
            "bits_presentes_por_mol_max_mediana": float(np.median(pres_mol_max)),
            "n_present_L2_por_espectro_media": float(diag["n_present_L2"].mean()),
            "n_present_L1_por_espectro_media": float(diag["n_present_L1"].mean()),
            "n_capture_por_espectro_media": float(diag["n_capture"].mean()),
            "mol_sin_ningun_bit_presente": int((pres_mol == 0).sum()),
            "mol_sin_ningun_bit_presente_max": int((pres_mol_max == 0).sum()),
            "n_picos_media": float(diag["n_picos"].mean()),
            "corr_n_picos_n_present_L2": float(np.corrcoef(
                diag["n_picos"], diag["n_present_L2"])[0, 1]),
        }
        log(f"P_{tag}_topo: {Pm.shape[0]:,} moléculas, pred media {Pm.mean():.4f}, "
            f"bits presentes/mol media {pres_mol.mean():.1f} (mediana {np.median(pres_mol):.0f}), "
            f"max-agreg media {pres_mol_max.mean():.1f} (mediana {np.median(pres_mol_max):.0f}), "
            f"L2/espectro {diag['n_present_L2'].mean():.1f}, "
            f"L1/espectro {diag['n_present_L1'].mean():.1f}, "
            f"capturas/espectro {diag['n_capture'].mean():.1f}, "
            f"sin bit presente: {(pres_mol == 0).sum()} (max: {(pres_mol_max == 0).sum()}), "
            f"corr riqueza-presentes {np.corrcoef(diag['n_picos'], diag['n_present_L2'])[0, 1]:.3f}")
    (VERIF / f"topo_predecir{args.sufijo}.json").write_text(json.dumps(out, indent=2))
    log(f"FIN predecir topo: {out}")


if __name__ == "__main__":
    main()
