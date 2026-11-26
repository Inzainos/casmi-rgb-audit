"""Validación de equivalencia de huellas Morgan entre versiones de RDKit.
CASMI paso 2 — validar_huellas_rdkit.py

Compara las huellas Morgan (2048 bits, radio 2) generadas con el RDKit
pinned del repo (2024.03.6, Python del sistema) contra las generadas con
RDKit 2026.9.1 (py313-venv), sobre la misma muestra de 500 SMILES del pool.
Replica EXACTAMENTE el código de 04_huellas.py:
  gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
  mol = Chem.MolFromSmiles(sm); fp = gen.GetFingerprint(mol)

Uso:
  python3 validar_huellas_rdkit.py --salida verificacion/huellas_muestra_rdkit2024.03.6.npz
  ~/workspace/py313-venv/bin/python validar_huellas_rdkit.py --salida verificacion/huellas_muestra_rdkit2026.9.1.npz
La comparación (bits distintos, moléculas distintas, fallos de parseo,
y consistencia del pool huellas_pool.npz con 2024.03.6) se hace aparte
con numpy y se registra en verificacion/validacion_rdkit.json.

Reglas: nada a Kaggle/GitHub; log en verificacion/; BASE = enveda-casmi-2026.
"""
import argparse, time, resource
from pathlib import Path
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

T0 = time.time()
def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time()-T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)

BASE = Path("/home/deamon/wt_deamonx/CASMI-sandbox/pruebas")  # PARCHE SANDBOX KALI 2026-11-10 (era /home/hatch/...)
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"
NBITS = 2048
RADIO = 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", default=str(INTER / "muestra_smiles_validacion_rdkit.txt"))
    ap.add_argument("--salida", required=True)
    args = ap.parse_args()

    smiles = [l.strip() for l in Path(args.muestra).read_text().splitlines() if l.strip()]
    log(f"RDKit {rdkit.__version__}: {len(smiles):,} SMILES de muestra")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIO, fpSize=NBITS)
    fps = np.zeros((len(smiles), NBITS), dtype=bool)
    fallos = []
    for i, sm in enumerate(smiles):
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            fallos.append(i)
            continue
        fp = gen.GetFingerprint(mol)
        fps[i, list(fp.GetOnBits())] = True
    np.savez(args.salida, fp=fps,
             version=np.array([rdkit.__version__]),
             fallos=np.array(fallos, dtype=np.int64),
             n_muestra=np.array([len(smiles)]))
    log(f"FIN: {len(smiles):,} huellas, {len(fallos)} fallos de parseo, "
        f"bits/mol media {fps.sum(axis=1).mean():.1f} -> {args.salida}")

if __name__ == "__main__":
    main()
