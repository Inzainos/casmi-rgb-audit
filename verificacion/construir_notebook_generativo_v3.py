#!/usr/bin/env python3
"""Construye verificacion/casmi_notebook_generativo_v3_u01.ipynb a partir
del notebook v2 regenerado por el constructor.

Método (precedente u06): el notebook v2 se construye siempre desde el
constructor (`verificacion/construir_notebook_generativo.py`), nunca a mano;
la variante se obtiene parchando el notebook generado con un script (no a
mano en un editor), con aserciones de texto exacto: si el texto esperado no
está, el script falla sin escribir nada.

Deltas v3 (todos verificados contra el diseño y la adaptación V3):
- G-V6 (igual que u06): UMBRAL_COS default "0.95" -> "0.6" (celda 1).
- v3: el checkpoint re-entrenado con aumento por corrupción
  (Kali 2026-11-19, 20/20 épocas, loss 1.7793 -> 0.9734, p_corrupt 0.5,
  t_corrupt 0.31, perfil bin05, md5 cfe4f3696399959500cc054ceb259184,
  33,947,033 B) vive en verificacion/checkpoint_v3/modelo.pt. Su
  directorio se antepone a LOCAL_GEN en la celda 1 para que la corrida
  local cargue el v3; en Kaggle /kaggle/input se busca primero, así que
  ahí manda el modelo.pt del dataset (que en el dataset v3 es el mismo
  checkpoint). La binarización se queda en 0.5 (la sonda del paso 3 fue
  NO_ADOPTAR); BEAM 25, blindaje y fusión SIN CAMBIO; portón congelado.
- Procedencia del checkpoint actualizada en CODE6 (celda 6), MD1 (celda 0)
  y MD2 (celda 9). Los textos de MD1/MD2 que decían umbral 0.95 se
  corrigen a 0.6 (en u06 quedaron con el 0.95 viejo aunque el código usa
  0.6; aquí el documento coincide con el código).

El harness G4_V3 NO pasó el criterio 0.021 (novel 0.0115, IC95
[0.0055,0.0185], adopta_checkpoint_v3: false); este build va por la
decisión de Elán 2026-11-21 04:04 ("aplica la vía generativa") + su GO
explícito 11:38 ("si ya se puede hacer y ya se tiene, no sabe por qué no
han avanzado"), que overrides el portón del GO condicional 2026-11-19.
El build es empaquetado + verificación local; NO re-mide ni reclama
mejora. Nada a Kaggle sin GO aparte.

Uso (desde la raíz del repo):
  python3 verificacion/construir_notebook_generativo.py   # regenera v2
  python3 verificacion/construir_notebook_generativo_v3.py
  -> escribe verificacion/casmi_notebook_generativo_v3_u01.ipynb
"""
import hashlib
import json
from pathlib import Path

REPO = Path("/home/hatch/workspace/enveda-casmi-2026")
NB_V2 = REPO / "verificacion" / "casmi_notebook_generativo_v2.ipynb"
NB_V3 = REPO / "verificacion" / "casmi_notebook_generativo_v3_u01.ipynb"
CKPT_V3 = REPO / "verificacion" / "checkpoint_v3" / "modelo.pt"
CKPT_V3_MD5 = "cfe4f3696399959500cc054ceb259184"

# --- Verificación previa: el checkpoint v3 local es el esperado ---
h = hashlib.md5()
with open(CKPT_V3, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
assert h.hexdigest() == CKPT_V3_MD5, \
    f"checkpoint v3 local md5 {h.hexdigest()} != {CKPT_V3_MD5}"
assert CKPT_V3.stat().st_size == 33_947_033, \
    f"checkpoint v3 tamaño {CKPT_V3.stat().st_size} != 33,947,033"
print(f"checkpoint v3 verificado: {CKPT_V3} ({CKPT_V3.stat().st_size:,} B, "
      f"md5 {CKPT_V3_MD5})")

nb = json.loads(NB_V2.read_text())
assert len(nb["cells"]) == 10, f"el v2 trae {len(nb['cells'])} celdas, no 10"

def src(i):
    return "".join(nb["cells"][i]["source"])

def set_src(i, nuevo):
    # nbformat guarda el source como lista de líneas con \n final.
    nb["cells"][i]["source"] = nuevo.splitlines(keepends=True)

def parche(i, viejo, nuevo, etiqueta):
    s = src(i)
    assert s.count(viejo) == 1, \
        f"celda {i} ({etiqueta}): texto esperado aparece {s.count(viejo)} veces, no 1"
    set_src(i, s.replace(viejo, nuevo, 1))
    print(f"celda {i}: aplicado [{etiqueta}]")

# --- (1) Delta G-V6: UMBRAL_COS default 0.95 -> 0.6 (celda 1) ---
parche(1,
       'UMBRAL_COS = float(os.environ.get("CASMI_UMBRAL_COS", "0.95"))',
       'UMBRAL_COS = float(os.environ.get("CASMI_UMBRAL_COS", "0.6"))',
       "G-V6 UMBRAL_COS 0.6")

# --- (2) v3: anteponer el directorio del checkpoint v3 a LOCAL_GEN (celda 1) ---
# Se inserta justo después del cierre de la lista LOCAL_GEN.
parche(1,
       "    Path('verificacion'),\n]\n",
       "    Path('verificacion'),\n]\n"
       "# v3 (2026-11-21, GO explícito de Elán): el checkpoint re-entrenado con\n"
       "# aumento por corrupción vive en verificacion/checkpoint_v3/modelo.pt\n"
       "# (md5 cfe4f3696399959500cc054ceb259184, 33,947,033 B). Se antepone su\n"
       "# directorio a LOCAL_GEN para que la corrida local cargue el v3; en\n"
       "# Kaggle /kaggle/input se busca primero, así que ahí manda el\n"
       "# modelo.pt del dataset (que en el dataset v3 es el mismo checkpoint).\n"
       "LOCAL_GEN = [Path('/home/hatch/workspace/enveda-casmi-2026/verificacion/checkpoint_v3')] + LOCAL_GEN\n",
       "v3 LOCAL_GEN checkpoint_v3 primero")

# --- (3) CODE6: procedencia del checkpoint (celda 6) ---
parche(6,
       "# El checkpoint es el re-entrenado (Colab 2026-11-13 20:45, beam exact-match\n"
       "# 0.2525, greedy 0.2335). La entrada es la huella BINARIA (umbral 0.5).",
       "# El checkpoint es el v3 re-entrenado con aumento por corrupción (Kali,\n"
       "# 2026-11-19, 20/20 épocas, loss 1.7793 -> 0.9734, p_corrupt 0.5,\n"
       "# t_corrupt 0.31, perfil bin05, md5 cfe4f3696399959500cc054ceb259184).\n"
       "# La entrada es la huella BINARIA (umbral 0.5).",
       "CODE6 procedencia checkpoint v3")

# --- (4) MD1: procedencia + umbral (celda 0) ---
parche(0,
       "decodificador Transformer huella→SMILES (beam-25, checkpoint del piloto B re-entrenado). "
       "Si el top-1 coseno del retrieval es ≥ 0.95,",
       "decodificador Transformer huella→SMILES (beam-25, checkpoint v3 re-entrenado con aumento "
       "por corrupción, Kali 2026-11-19, md5 cfe4f369…). "
       "Si el top-1 coseno del retrieval es ≥ 0.6,",
       "MD1 procedencia checkpoint v3 + umbral 0.6")

# --- (5) MD2: procedencia en Datos + umbral en Híbrido (celda 9) ---
parche(9,
       "el checkpoint del piloto B (trabajo propio, re-entrenado en Colab 2026-11-13),",
       "el checkpoint v3 (trabajo propio, re-entrenado en la Kali 2026-11-19 con aumento por "
       "corrupción, md5 cfe4f369…),",
       "MD2 Datos procedencia checkpoint v3")
parche(9,
       "retrieval-primero con umbral top-1 coseno 0.95 (protocolo §6).",
       "retrieval-primero con umbral top-1 coseno 0.6 (protocolo §6, re-calibrado componente 3, "
       "portón congelado).",
       "MD2 Híbrido umbral 0.6")

nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.12.3"}
with open(NB_V3, "w") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
md5 = hashlib.md5(NB_V3.read_bytes()).hexdigest()
print(f"notebook v3 escrito: {NB_V3} ({len(nb['cells'])} celdas, "
      f"{sum(1 for c in nb['cells'] if c['cell_type'] == 'code')} de código)")
print(f"md5 notebook v3: {md5} | tamaño: {NB_V3.stat().st_size:,} B")
