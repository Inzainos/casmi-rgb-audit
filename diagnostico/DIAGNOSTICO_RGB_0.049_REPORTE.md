# Reporte: por qué el RGB local 0.4951 salió en 0.049 público

Orden de Elán (2026-11-11, vía Coordinador): investigar qué sucedió con el
envío RGB y crear este reporte.
Fecha del reporte: 2026-11-11 ~21:15 CST. Autor: Mole (chat experto CASMI).
Archivo: `verificacion/DIAGNOSTICO_RGB_0.049_REPORTE.md`.
Trabajo previo relacionado: `verificacion/DIAGNOSTICO_BRECHA_RGB.md`
(2026-11-11 ~21:10, la diagnosis corta que lo precede; este reporte es el
entregable formal con las tres hipótesis probadas una por una).

## Pregunta central

¿Por qué el RGB local 0.4951 (MRR@25 mol-split) salió en **0.049 público**,
10x menos, y por debajo del **0.146** del envío 3 (baseline 1)?

## Hechos verificados (no supuestos)

- Envío RGB: ref **58256941**, "Notebook casmi-runtime-notebook-rgb |
  Version 6", **2026-11-11 19:40:47 CST**, status **COMPLETE**,
  public_score **0.049**, submission.csv **457,663 B**,
  scriptVersionId **368622826**. Kernel:
  https://www.kaggle.com/code/elnzainoscorona/casmi-runtime-notebook-rgb
  (verificado por mí con `get_competition_submission` vía conector
  custom.kaggle: ref, bytes, fecha, descripción, puntaje y estado coinciden).
- Local: rgb20k_bin MRR@25 mol-split **0.4951** (RESULTADOS.md, ronda 2,
  n=13,748); submission local `salida/submission_rgb.csv` md5
  **6dcceab5d57a00b82d485439ebd6061b**, 400×25, 10,000/10,000 SMILES válidos
  (md5 re-verificado por mí en esta sesión).
- Notebook: `casmi_runtime_notebook_rgb.ipynb` md5
  **1de7b37f88f7afe1577d44fc28b7542d**, 7 celdas (md5 re-verificado).
  Dataset V2 Ready, 6 archivos, W_lineal_rgb20k.npy **491,520,128 B**
  incluida (verificado por mí con `list_dataset_files` V2: los 6 archivos
  con sus bytes exactos).
- Competencia: enveda-CASMI26-molecule-id-mass-spectra.
- Envío 3 (baseline 1, mol-split local 0.0000): **0.146 público**.
  El test oculto trae clases 2 y 3 (f_class1 ≈ 0.17).
- Selección final: **0/2**, sin tocar. Nada reenviado.

---

## Hipótesis 1 — El notebook en runtime NO usó el modelo RGB/W como debía

**Veredicto: REFUTADA (con una limitación explícita, ver §1.4).**

### 1.1 Los insumos del runtime estaban correctos

- Dataset V2 verificado por mí vía conector (`list_dataset_files`,
  versionNumber 2): W_lineal_rgb20k.npy 491,520,128 B;
  bins_rgb.npy 80,128 B; huellas_pool.npz 17,768,378 B;
  mapa_smiles_full.parquet 10,083,846 B; mol_index.npy 6,619,552 B;
  precursores_full_idx.npz 29,257,268 B. Total 555,329,300 B. Status Ready.
- El notebook local (md5 1de7b37f…) reproduce `salida/submission_rgb.csv`
  byte-idéntico en corrida local (4.3 s, RSS 658 MB, 7/7 checks OK;
  verificación del 2026-11-10, `verificacion/runtime_rgb.json`).

### 1.2 La comparación Kaggle-vs-local que SÍ se pudo hacer

La prueba clave pedida era bajar el submission.csv de Kaggle (457,663 B) y
compararlo contra el local. **Esa comparación directa NO se pudo hacer por
la vía programática**: `download_competition_submission` (ref 58256941)
devuelve "Permission 'submissions.download' was denied" (negación dura del
permiso; se detuvo ahí, no se buscó rodeo). Ver §1.4.

Lo que SÍ se pudo bajar es la salida del kernel v6
(`download_notebook_output`, kernel casmi-runtime-notebook-rgb, v6,
submission.csv, vía URL firmada): es la corrida del workspace sobre el
test VISIBLE, y es la comparación válida del runtime contra local sobre
los MISMOS datos:

| Medida | Local (salida/submission_rgb.csv) | Kernel v6 en Kaggle |
|---|---|---|
| Tamaño | 396,496 B | 396,528 B |
| Filas × SMILES | 400 × 25 | 400 × 25 |
| molecule_id | 400 (test visible) | 400, idénticos en orden |
| SMILES distintos | 9,183 | 9,178 |
| SMILES distintos compartidos | — | 9,050 / 9,183 (98.6%) |
| Filas vacías / con <25 | 0 / 0 | 0 / 0 |
| Filas degeneradas (top-1 ×25) | 0 | 0 |
| **MRR@25 en test visible** | **0.5758** (top-1 0.4550, recall@25 0.9175) | **0.5723** (top-1 0.4500, recall@25 0.9200) |

Archivo descargado: `verificacion/submission_kaggle_rgb_58256941.csv`
(396,528 B, md5 13cb3ebf8a43b4ef0943e103c3d8d0e9).

El runtime de Kaggle reprodujo el cómputo local esencialmente exacto
(0.5723 vs 0.5758; la diferencia de 32 B son 1–2 rankings por
nondeterminismo numérico entre runtimes, no degradación). **W sí se usó;
el modelo corrió completo; no hubo fallback ni salida degenerada.**

### 1.3 El archivo enviado (457,663 B) es consistente con una salida sana

El archivo de la re-corrida oculta es 61,167 B (+15%) más grande que la
salida visible: 45.8 vs 39.7 bytes por SMILES en promedio. Eso es
consistente con moléculas ocultas distintas y en promedio más grandes, no
con un fallback (una salida "CCO" o de top-1 repetido sería mucho más
chica y/o degenerada). No es prueba del contenido, es evidencia de tamaño:
no hay señal de degradación.

### 1.4 Limitación explícita

La comparación directa del archivo ENVIADO contra el local NO se completó
porque el permiso `submissions.download` del token MCP lo niega. La
comparación que sustituye (kernel v6 vs local, mismos datos) refuta la
degradación del runtime, pero la estructura del archivo oculto (SMILES
distintos, patrón de relleno por fila, si todos los SMILES son del pool)
sigue sin medirse. Verificación pendiente: bajarlo por el navegador desde
la página de envíos (es el propio envío de Elán) y repetir esta tabla
sobre ese archivo.

---

## Hipótesis 2 — La validación local 0.4951 está inflada

**Veredicto: CONFIRMADA en el sentido preciso: el número es correcto para
lo que mide, pero lo que mide no es lo que el test oculto premia. La
inflación es estructural (~6x), no un error de cómputo.**

### 2.1 El número 0.4951 está verificado desde su artefacto primario

Leído por mí del JSON de métricas en la Kali
(`pruebas/verificacion/metricas_masa_rgb20k_bin_val_mol.json`):

| Campo | Valor |
|---|---|
| tag / split / bin | rgb20k_bin / val_mol / true |
| tol_ppm / tol_min | 10.0 / 0.002 |
| n | 13,748 |
| cand_mediana / cand_media | 106 / 141.03 |
| **mrr25** | **0.4951** |
| top1 / top5 / recall25 | 0.371327 / 0.645839 / 0.850596 |
| rank_mediana_presentes | 2.0 |

Crudo (mismo archivo, bin=false): mrr25 0.45419, top1 0.326666,
top5 0.608234, recall25 0.830084. Coincide con RESULTADOS.md
(`pruebas/RESULTADOS.md`, ronda 2). El cómputo usa `mrr_from_ranks`
de `casmi_lib` (nunca promedia 1/rango solo sobre hits; ese bug se
detectó y corrigió en la ronda 2, `22_rerank.py`).

### 2.2 La inflación: el pool contiene la verdad del 100% de las consultas

Medido por mí desde los archivos (no supuesto):

- Pool de retrieval: `huellas_pool.npz` shape **(275,810, 2,048)**.
  `pool == mols` exacto: son TODAS las moléculas de train.parquet.
- Consultas de validación: **val_mols 13,790/13,790 dentro del pool**;
  spec_mols 9,943/9,943 dentro del pool. Cero fuera.
  (Fuente: `verificacion/split_arrays.npz` vs
  `runtime_rgb_dataset/mol_index.npy`.)
- La evaluación lee `Y_true = pool[q_mol]` (15_evaluar_pred.py:55): la
  verdad está en el pool para el 100% de las consultas val. El oráculo
  0.9937 de la fase Morgan lo confirma por otro camino.
- El modelo se entrenó con 261,536 moléculas (1 espectro c/u, de val_lib)
  y se validó con las 13,748 restantes. Nunca vio los ESPECTROS de val en
  entrenamiento, pero SÍ puede recuperar sus HUELLAS del pool: el diseño
  mide "¿el modelo generaliza a espectros nuevos?", no "¿resuelve
  moléculas nuevas?".

### 2.3 Corroboración independiente del rendimiento local

El mismo modelo, puntuado por mí contra la verdad del test visible
(400 moléculas, todas clase-1; verdad = inchikey14 del top-1 de baseline-1,
coseno 0.9999999956, 400/400 ≥ 0.95, verificado en paso 2; mapeo por
`mapa_smiles_full.parquet`, 400/400, 0 SMILES fuera del pool):

- RGB local: **MRR@25 0.5758** (top-1 0.4550, recall@25 0.9175).
- RGB kernel v6: **0.5723**.

El modelo es genuinamente bueno en retrieval de clase-1 (0.50–0.58).
El problema no es el número; es que el test oculto no es clase-1 en su
mayoría.

### 2.4 Cuánto infla

El mol-split simula un test donde el 100% de las moléculas tiene su
verdad en el pool. El oculto tiene ~17% así. El factor de inflación es
~1/0.17 ≈ 6x: 0.4951 local corresponde a ~0.08 esperado en el oculto
(ver §3), no a 0.4951. La métrica correcta para ordenar modelos de cara
al oculto es evaluar con el pool SIN las moléculas de consulta
(simulación novedosa) o reportar MRR clase-1 × f_class1.

---

## Hipótesis 3 — El test oculto es genuinamente mucho más duro

**Veredicto: CONFIRMADA como mecanismo principal (composición), con un
residuo real secundario que no queda totalmente explicado.**

### 3.1 Cuantificación: lo que se esperaría del RGB en el oculto

- f_class1 ≈ 0.146 / 0.8691 = **0.168** (público de baseline-1 ÷ su
  spec-split local 0.8691; la fase Morgan estimó ≈0.17 por el mismo camino).
- Público esperado del RGB:
  - con val 0.4951: 0.168 × 0.4951 = **0.083**;
  - con test visible 0.5758: 0.168 × 0.5758 = **0.097**.
- Observado: **0.049**.

### 3.2 El residuo

- Rendimiento RGB implicado en clase-1 oculta: 0.049 / 0.168 = **0.292**,
  contra 0.4951 (val) y 0.5758 (test visible).
- Cotas: aunque el rendimiento clase-1 oculta de baseline-1 fuera
  0.70–1.00 (f_class1 0.146–0.209), el público esperado del RGB sería
  0.084–0.120. El 0.049 queda por debajo de todo el rango.
- Ruido: ~67 moléculas puntuables en el oculto; SE del MRR público
  ≈ 0.008. El faltante (0.034–0.048) es ~4–6 SE: **real, no ruido**.
- Causas candidatas del residuo (no separadas por las mediciones
  actuales): las moléculas ocultas son más grandes (sus SMILES promedian
  45.8 vs 39.7 bytes en la salida visible) y/o la clase-1 oculta se
  concentra en librerías más duras (el desglose por librería/tamaño del
  modelo RGB nunca se midió; en baseline-1, enveda-180 —la más cercana al
  test— es la más dura con 0.8413 spec-split).
- La lógica de resolución es idéntica en local y en Kaggle (misma W,
  mismo pool, mismo código, agregación por media de espectros por
  molécula — 23_predecir_test_rgb.py:13,120; verificada por la
  reproducción del workspace §1.2). La diferencia está en el TEST, no
  en el runtime.
- Lo que separaría las causas del residuo: la estructura del archivo
  enviado (§1.4) o un desglose por tamaño/librería del RGB en val
  (archivos P_val en la Kali, no medido).

### 3.3 ¿El 0.049 es anómalo?

No como orden de magnitud: un retrieval de novo en este oculto DEBE
puntuar ~0.08–0.10, no 0.4951. El 0.049 está por debajo de esa banda
esperada, pero dentro del mismo orden: es el 0.08–0.10 esperado
degradado por una clase-1 oculta más dura (~0.29 vs ~0.50). Anómalo
sería 0.49 o 0.00; 0.049 es un "peor de lo esperado" medible, no un
fallo catastrófico del envío.

---

## Veredicto (cuál hipótesis explica la brecha)

**H2 + H3, en ese orden. H1 refutada.**

1. **H2 (diseño de la validación) explica la mayor parte**: el 0.4951
   mide retrieval con la verdad en el pool (100% de las consultas val);
   el oculto es ~83% novedoso, donde un retrieval puntúa 0 por
   construcción. Eso solo lleva el 0.4951 a ~0.083–0.097 esperados.
2. **H3 (composición + dureza del oculto) explica el resto**: la clase-1
   oculta rinde ~0.29 para el RGB contra ~0.50–0.58 local, un residuo
   real de 0.034–0.048 que no es ruido.
3. **H1 (runtime degradado) queda refutada** por la reproducción del
   kernel v6 (0.5723 vs 0.5758 local) y por los insumos verificados;
   con la limitación de §1.4 (el archivo oculto en sí no se pudo bajar).

## Conclusión: ¿el RGB tiene arreglo? ¿O el camino es otro?

- **El modelo RGB está bien; no tiene arreglo dentro del retrieval.**
  Su trabajo (recuperar la huella correcta cuando está en el pool) lo
  hace a 0.50–0.58. Ningún ajuste del lineal RGB, del K, del momentum o
  de la CE puede superar el techo duro de un retrieval en este oculto:
  f_class1 × (rendimiento clase-1) ≈ **0.08–0.10**. El 0.049 < 0.146 lo
  confirma en la práctica.
- **El camino Morgan/blend tiene el mismo techo.** El lineal Morgan
  (0.4146) y el RGB (0.4951) son ambos retrieval; el blend Morgan+baseline-1
  (0.462 local) es un número mol-split con la misma inflación por
  verdad-en-pool. Su expectativa oculta es, en el mejor caso,
  f_class1 × (rendimiento clase-1 del blend) ≈ 0.17 × ~0.9 ≈ **0.15**,
  apenas sobre el 0.146, y solo si el blend supera de verdad a baseline-1
  en clase-1 (no medido en clase-1 sola). No es una victoria clara; antes
  de enviarlo hay que evaluarlo en clase-1 sola (pool sin consultas).
- **Para puntuar en clases 2/3 hay que GENERAR estructuras nuevas**, no
  recuperar conocidas. Eso es otro proyecto (modelo generativo
  espectro→SMILES), fuera del pipeline actual. Con el pipeline actual, el
  mejor resultado alcanzable en el oculto es la masa clase-1, que
  baseline-1 ya captura a ~0.87.
- **Recomendación medida (decisión de Elán, no de este reporte):** la
  selección final (2 slots, hoy 0/2) debería salir de los envíos con
  puntaje público real — hoy solo el envío 3 (0.146) lo tiene — y no de
  los MRR locales inflados. Cualquier envío nuevo (Morgan, blend) debe
  evaluarse primero en clase-1 sola con el pool sin consultas, o se
  repetirá la sorpresa.

## Verificaciones pendientes (no hechas, con su valor)

1. Bajar el submission.csv enviado (457,663 B) por el navegador desde la
   página de envíos y repetir la tabla de §1.2 sobre ESE archivo
   (SMILES distintos, patrón de relleno por fila, todos del pool).
   Valor: cierra §1.4 y separa el residuo de §3.2.
2. Desglose por tamaño de precursor / librería del RGB en val (P_val en
   la Kali). Valor: separa las causas del residuo de §3.2.
3. Evaluación clase-1 sola (pool sin consultas) de Morgan y del blend
   antes de cualquier envío nuevo. Valor: evita repetir la sorpresa.

## Archivos y logs citados

- Este reporte: `verificacion/DIAGNOSTICO_RGB_0.049_REPORTE.md`.
- Diagnosis corta previa: `verificacion/DIAGNOSTICO_BRECHA_RGB.md`.
- Salida del kernel v6 descargada:
  `verificacion/submission_kaggle_rgb_58256941.csv` (396,528 B,
  md5 13cb3ebf8a43b4ef0943e103c3d8d0e9).
- Métricas RGB val (Kali):
  `pruebas/verificacion/metricas_masa_rgb20k_bin_val_mol.json`
  (mrr25 0.4951, n=13,748) y `metricas_masa_rgb20k_val_mol.json`
  (crudo 0.45419); `pruebas/RESULTADOS.md` (ronda 2).
- Llamadas del conector:
  `~/workspace/custom-connectors/custom.kaggle/calls/get_submission_rgb.json`
  (ref 58256941: 457,663 B, 0.049, COMPLETE),
  `download_submission_rgb.json` (permiso denegado),
  `download_nb_output_rgb.json` (URL firmada de la salida v6),
  `list_dataset_files_v2.json` (dataset V2, 6 archivos, W presente).
- `download_competition_submission` se agregó a ALLOWED_TOOLS en
  `~/workspace/custom-connectors/custom.kaggle/bin/client.py` tras leer
  su esquema (regla del CONNECTOR.md); su uso dio permiso denegado.
- Submissions locales intactos (md5 verificados en esta sesión):
  submission.csv 7a7da389…, submission_paso2.csv ed758b99…,
  submission_relleno25.csv 7f082916…, submission_morgan.csv 85458472…,
  submission_rgb.csv 6dcceab5… (AGENTS.md regla 6).
- Nada marcado como final; nada reenviado. Solo lecturas a Kaggle.
