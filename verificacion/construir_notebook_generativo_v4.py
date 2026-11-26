#!/usr/bin/env python3
"""Construye verificacion/casmi_notebook_generativo_v4_u01.ipynb a partir
del notebook v3 (casmi_notebook_generativo_v3_u01.ipynb).

Método (precedente v3): el notebook se construye siempre desde un
constructor (nunca a mano en un editor), parchando el notebook base con
un script y aserciones de texto exacto: si el texto esperado no está
exactamente una vez, el script falla sin escribir nada.

Delta v4 (fusión V1, k=15, propuesta de optimización §2.2):
- La v4 es el notebook v3 con la fusión V1 (beam inyectado en la rama
  retrieval, k=15) en lugar de la V0. La fusión es el ÚNICO cambio
  funcional. El checkpoint sigue siendo el v3 (modelo.pt md5
  cfe4f3696399959500cc054ceb259184, 33,947,033 B); el portón sigue en
  0.6 congelado; BEAM 25, blindaje y UMBRAL_BIN sin cambio.
- V1 (medida en medir_fusion_v1_extension.py, md5 7dfadbe8…; propuesta
  §2.2 l.99-111, 125, 225): en la rama retrieval (top1_cos >= 0.6) la
  salida es ret top-(25-k) + beam top-k, k=15 (ret top-10 + beam
  top-15). En la rama generativa (top1_cos < 0.6) es igual que V0
  (beam primero, ret rellena). Números esperados: novel 0.0120860
  (+5 moléculas rescatadas), clase-1 Δ=0 exacto, híbrido 0.1577244
  (la mejor fusión medida). La propuesta dice: "V1 k=15 es la fusión
  a adoptar SI se construye una v3" (l.125) — pero la v3 se construyó
  con V0 (el diff v3 vs u06 fueron solo 4 cambios de proveniencia /
  checkpoint; la fusión no se tocó). V1 nunca se construyó en
  notebook. Esta es la v4.
- POR QUÉ LA V4 TOCA TAMBIÉN LA CELDA 6 (generación del beam): en la
  v3 el beam solo se genera para las moléculas con top1_cos < 0.6
  (gen_mols). Pero V1 inyecta el beam en la rama retrieval (top1_cos
  >= 0.6), donde en la v3 gen está vacío. Sin generar el beam para
  esas moléculas, V1 no tendría efecto (ret[:10] + gen vacío = solo
  ret). Por eso la celda 6 genera el beam para TODAS las moléculas
  (cuando GENERATIVO_ACTIVO). Esto es parte del cambio de fusión,
  no un cambio aparte: el beam, el checkpoint y el portón son los
  mismos; solo cambia PARA QUÉ moléculas se genera el beam.
- ROBUSTEZ (blindaje): en la rama retrieval, después de ret top-10
  + beam top-15 se agrega ret[10:] como fallback. Con beam completo
  y sin duplicados, ret[10:] nunca aparece en los primeros 25 (la
  salida es idéntica a la V1 medida: ret top-10 + beam top-15). El
  fallback solo actúa cuando el beam está corto o vacío (blindaje
  sin RDKit: gen=[] y la salida vuelve a ser ret top-25, igual que
  V0 — el blindaje sigue dando retrieval puro) o cuando hay
  duplicados por InChIKey14 (en vez de rellenar repitiendo el top-1,
  se completa con candidatos reales de ret). Sin este fallback, el
  blindaje se rompería (ret top-10 + 15 repeticiones del top-1).

GO: Elán 2026-11-24 11:57 (vía Coordinador): "Construyan la versión
cuatro con la versión uno". El GO cubre construir + enviar (después
de verificar). Nada a Kaggle sin verificación (el envío es un paso
separado, por el navegador).

Uso (desde la raíz del repo):
  python3 verificacion/construir_notebook_generativo_v4.py
  -> escribe verificacion/casmi_notebook_generativo_v4_u01.ipynb
"""
import hashlib
import json
from pathlib import Path

REPO = Path("/home/hatch/workspace/enveda-casmi-2026")
NB_V3 = REPO / "verificacion" / "casmi_notebook_generativo_v3_u01.ipynb"
NB_V4 = REPO / "verificacion" / "casmi_notebook_generativo_v4_u01.ipynb"
NB_V3_MD5 = "eb070f56de4c03911c617201571bcc69"
CKPT_V3 = REPO / "verificacion" / "checkpoint_v3" / "modelo.pt"
CKPT_V3_MD5 = "cfe4f3696399959500cc054ceb259184"

# --- Verificación previa: el notebook v3 y el checkpoint v3 son los esperados ---
assert hashlib.md5(NB_V3.read_bytes()).hexdigest() == NB_V3_MD5, \
    f"notebook v3 md5 {hashlib.md5(NB_V3.read_bytes()).hexdigest()} != {NB_V3_MD5}"
h = hashlib.md5()
with open(CKPT_V3, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
assert h.hexdigest() == CKPT_V3_MD5, \
    f"checkpoint v3 local md5 {h.hexdigest()} != {CKPT_V3_MD5}"
assert CKPT_V3.stat().st_size == 33_947_033, \
    f"checkpoint v3 tamaño {CKPT_V3.stat().st_size} != 33,947,033"
print(f"base v3 verificada: {NB_V3} (md5 {NB_V3_MD5})")
print(f"checkpoint v3 verificado: {CKPT_V3} ({CKPT_V3.stat().st_size:,} B, "
      f"md5 {CKPT_V3_MD5})")

nb = json.loads(NB_V3.read_text())
assert len(nb["cells"]) == 10, f"el v3 trae {len(nb['cells'])} celdas, no 10"

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

# --- (1) Celda 6: generar el beam para TODAS las moléculas (V1 lo necesita
# en la rama retrieval). Antes: solo top1_cos < UMBRAL_COS. ---
parche(6,
       "# Solo se genera para las moléculas que lo necesitan (top1_cos < UMBRAL_COS).",
       "# v4 (fusión V1, k=15): se genera el beam para TODAS las moléculas (no solo\n"
       "# top1_cos < UMBRAL_COS), porque V1 inyecta el beam top-15 en la rama\n"
       "# retrieval (top1_cos >= 0.6) y necesita los candidatos del beam ahí.\n"
       "# Sin esto, V1 no tendría efecto (gen estaría vacío en la rama retrieval).",
       "V1 celda 6 comentario beam para todas")
parche(6,
       "gen_mols = ([mid for mid in todos if ret_top1_cos[mid] < UMBRAL_COS]\n"
       "            if GENERATIVO_ACTIVO else [])",
       "gen_mols = (list(todos) if GENERATIVO_ACTIVO else [])",
       "V1 celda 6 gen_mols todas")
parche(6,
       "log(f\"moléculas que necesitan generación (top1_cos < {UMBRAL_COS}): \"\n"
       "    f\"{len(gen_mols)}/{len(todos)}\")",
       "log(f\"moléculas con beam generado (fusión V1: todas): \"\n"
       "    f\"{len(gen_mols)}/{len(todos)}\")",
       "V1 celda 6 log beam todas")

# --- (2) Celda 7: definir K_V1 = 15 (fusión V1) ---
parche(7,
       "out_rows = []\nn_ret_only = 0",
       "# Fusión V1 (k=15, propuesta de optimización §2.2): inyección de beam en la\n"
       "# rama retrieval. En la rama retrieval (top1_cos >= 0.6): ret top-(25-k) +\n"
       "# beam top-k. En la rama generativa (top1_cos < 0.6): igual que V0 (beam\n"
       "# primero, ret rellena).\n"
       "K_V1 = 15\n"
       "out_rows = []\nn_ret_only = 0",
       "V1 celda 7 K_V1 = 15")

# --- (3) Celda 7: rama retrieval V0 -> V1 ---
parche(7,
       "    if ret_top1_cos.get(mid, 0.0) >= UMBRAL_COS:\n"
       "        for ik, sm in ret:\n"
       "            agregar(out, seen, sm, ik)\n"
       "        n_ret_only += 1",
       "    if ret_top1_cos.get(mid, 0.0) >= UMBRAL_COS:\n"
       "        # V1: ret top-(25-k) + beam top-k; si el beam está corto o vacío\n"
       "        # (blindaje sin RDKit), se completa con el resto de ret para no\n"
       "        # perder candidatos de retrieval (el blindaje sigue dando retrieval\n"
       "        # puro: sin RDKit gen=[] y la salida es ret top-25).\n"
       "        for ik, sm in ret[:25 - K_V1]:\n"
       "            agregar(out, seen, sm, ik)\n"
       "        for sm in gen[:K_V1]:\n"
       "            agregar(out, seen, sm)\n"
       "        for ik, sm in ret[25 - K_V1:]:\n"
       "            agregar(out, seen, sm, ik)\n"
       "        n_ret_only += 1",
       "V1 celda 7 rama retrieval ret+beam")

# --- (4) MD1 (celda 0): descripción de la fusión V1 ---
parche(0,
       "Si el top-1 coseno del retrieval es ≥ 0.6, el retrieval ocupa las 25 posiciones; "
       "si es menor, el generativo va primero y el retrieval rellena hasta 25.",
       "Si el top-1 coseno del retrieval es ≥ 0.6, la fusión V1 (k=15) toma ret top-10 + "
       "beam top-15 (el beam se genera para todas las moléculas); si es menor, el "
       "generativo va primero y el retrieval rellena hasta 25.",
       "MD1 fusión V1")

# --- (5) MD2 (celda 9): Híbrido V1 ---
parche(9,
       "- **Híbrido:** retrieval-primero con umbral top-1 coseno 0.6 (protocolo §6, re-calibrado componente 3, portón congelado). En el test visible todas las moléculas son clase-1 (top1_cos ~1.0), así que el generativo no dispara ahí; en el oculto (~17% clase-1 estimado) dispara en ~83%. La puerta G0 (formato) se verifica en el visible; la calidad del generativo con huella predicha (G3/G4) se mide en GEN-VAL con el evaluador, no en este notebook.",
       "- **Híbrido (fusión V1, k=15, propuesta de optimización §2.2):** retrieval-primero con umbral top-1 coseno 0.6 (protocolo §6, re-calibrado componente 3, portón congelado). En la rama retrieval (top1_cos ≥ 0.6) la salida es ret top-10 + beam top-15 (el beam se genera para todas las moléculas, no solo top1_cos < 0.6); en la rama generativa (top1_cos < 0.6) el generativo va primero y el retrieval rellena, igual que V0. En el test visible todas las moléculas son clase-1 (top1_cos ~1.0), así que todas toman la rama retrieval con beam inyectado; en el oculto (~17% clase-1 estimado) la rama generativa dispara en ~83%. La puerta G0 (formato) se verifica en el visible; la calidad del generativo con huella predicha (G3/G4) se mide en GEN-VAL con el evaluador, no en este notebook.",
       "MD2 Híbrido V1")

# --- (6) MD2 (celda 9): Acelerador (V4 genera beam para todas) ---
parche(9,
       "- **Acelerador:** GPU obligatoria (P100/T4). Medido en local (CPU, 4 hilos): beam-25 a ~96 s/molécula, así que 400 moléculas tardarían ~10.7 h en CPU y NO caben en el límite de 9 h. En GPU (T4, medido en Colab: 2,000 moléculas en 867 s) el total estimado para el test visible es ~15 min. CPU solo sirve para pruebas con pocas moléculas.",
       "- **Acelerador:** GPU obligatoria (P100/T4). Medido en local (CPU, 4 hilos): beam-25 a ~96 s/molécula, así que 400 moléculas tardarían ~10.7 h en CPU y NO caben en el límite de 9 h. En GPU (T4, medido en Colab: 2,000 moléculas en 867 s) el total estimado para el test visible es ~15 min. CPU solo sirve para pruebas con pocas moléculas. La v4 genera el beam para todas las moléculas (fusión V1), así que en CPU el costo es mayor que en la v3 (que solo generaba para top1_cos < 0.6); en GPU el total sigue estimado en ~15 min.",
       "MD2 Acelerador V4 beam para todas")

nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.12.3"}
with open(NB_V4, "w") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
md5 = hashlib.md5(NB_V4.read_bytes()).hexdigest()
print(f"notebook v4 escrito: {NB_V4} ({len(nb['cells'])} celdas, "
      f"{sum(1 for c in nb['cells'] if c['cell_type'] == 'code')} de código)")
print(f"md5 notebook v4: {md5} | tamaño: {NB_V4.stat().st_size:,} B")
