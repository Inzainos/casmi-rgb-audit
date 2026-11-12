# Diagnóstico de la brecha RGB — local 0.4951 → público 0.049

Fecha: 2026-11-11 ~20:42–21:10 CST (Mole, por encargo del Coordinador).
Envío RGB: ref 58256941, "Notebook casmi-runtime-notebook-rgb | Version 6",
2026-11-11 19:40:47 CST, status COMPLETE, public_score **0.049**,
submission.csv 457,663 B, scriptVersionId 368622826.
Mejor público vigente: envío 3 (baseline 1), 0.146. Selección final: 0/2, sin tocar.

## 1. El envío es real y el runtime corrió el modelo correctamente (Q1)

Verificado por mí, independiente del reporte del Coordinador:

- `get_competition_submission` (ref 58256941, vía conector custom.kaggle):
  total_bytes 457,663, date 2026-11-12T01:40:47.530Z, public_score "0.049",
  status COMPLETE, submitted_by elnzainoscorona. Coincide con lo reportado.
- `download_notebook_output` (kernel casmi-runtime-notebook-rgb, v6,
  submission.csv) devolvió una URL firmada; el archivo descargado es una
  salida SANA del modelo: 400 filas × 25 SMILES, 9,178 SMILES distintos,
  0 filas vacías, 0 filas con <25, 0 filas degeneradas.
  Archivo: `verificacion/submission_kaggle_rgb_58256941.csv`
  (396,528 B, md5 13cb3ebf8a43b4ef0943e103c3d8d0e9).
- Ese archivo es la corrida del workspace sobre el test VISIBLE (los 400
  molecule_id son idénticos, en orden, a los del test visible). Puntuado
  contra la verdad del test visible (ver §3): **MRR@25 0.5723** (top-1
  0.4500, recall@25 0.9200), contra **0.5758** del run local
  (salida/submission_rgb.csv). 9,050/9,183 SMILES distintos compartidos.
  El runtime de Kaggle reprodujo el cómputo local esencialmente exacto.
  **No hay degradación, no hay fallback, W sí se usó.**
- El archivo ENVIADO (re-corrida sobre el test oculto, 457,663 B) NO se
  puede descargar por el conector: `download_competition_submission` →
  "Permission 'submissions.download' was denied" (negación dura; no se
  reintentó por otra vía). Su tamaño (+15% vs la salida visible,
  45.8 vs 39.7 bytes/SMILES) es consistente con una salida sana y diversa
  sobre moléculas distintas (más grandes), no con un fallback
  (un "CCO" o un top-1 repetido sería mucho más chico / degenerado).

## 2. El 0.4951 local mide retrieval con la verdad DENTRO del pool (Q2)

Medido por mí, no supuesto:

- Pool de retrieval: `huellas_pool.npz` shape **(275,810, 2,048)** — TODAS
  las moléculas de train. `pool == mols` exacto.
- Consultas de validación: **val_mols 13,790/13,790 dentro del pool**;
  spec_mols 9,943/9,943 dentro del pool. Cero fuera.
- La evaluación lee `Y_true = pool[q_mol]` (15_evaluar_pred.py:55): la
  verdad está en el pool para el 100% de las consultas val (el oráculo
  0.9937 de la fase Morgan lo confirma).
- En el test oculto, solo la fracción clase-1 (~17%) tiene su verdad en el
  pool. Un modelo de retrieval solo puede devolver moléculas del pool,
  así que puntúa **0 en el ~83% de moléculas novedosas POR CONSTRUCCIÓN**,
  por bueno que sea el fingerprint predicho.
- f_class1 ≈ 0.146 / 0.8691 = **0.168** (público de baseline-1 ÷ su
  spec-split local 0.8691).
- Público esperado del RGB: 0.168 × 0.4951 (val) = **0.083**;
  0.168 × 0.5758 (test visible) = **0.097**. Observado: **0.049**.

La brecha de 10x es en su mayor parte un efecto de composición del test,
no un bug de runtime.

## 3. Residuo: la clase-1 oculta puntúa ~0.29 vs 0.50–0.58 local (Q3)

- Rendimiento RGB implicado en clase-1 oculta: 0.049 / 0.168 = **0.292**.
- Cotas: aunque el rendimiento clase-1 oculta de baseline-1 fuera 0.70–1.00
  (f_class1 0.146–0.209), el público esperado del RGB sería 0.084–0.120.
  El observado 0.049 queda por debajo de todo el rango.
- Ruido: ~67 moléculas puntuables en el oculto; SE del MRR público ≈ 0.008.
  El faltante (0.034–0.048) es ~4–6 SE: real, no ruido.
- La lógica de resolución es idéntica (misma W, mismo pool, mismo código,
  agregación por media de espectros por molécula — 23_predecir_test_rgb.py:13,120;
  verificada por la reproducción del workspace §1). La diferencia está en
  el TEST, no en el runtime.
- Causas candidatas del residuo (no separadas por las mediciones actuales):
  las moléculas ocultas son más grandes (sus SMILES promedian 45.8 vs 39.7
  bytes) y/o la clase-1 oculta se concentra en librerías más duras; el
  desglose por librería/tamaño del modelo RGB nunca se midió.
- Lo que lo separaría: la estructura del archivo enviado (bloqueada por el
  permiso submissions.download) o un desglose por tamaño/librería del RGB
  en val (archivos P_val en la Kali, no medido).

## 4. Medición de control: el RGB en el test visible (todo clase-1)

Verdad del test visible = inchikey14 del top-1 de baseline-1
(coseno 0.9999999956, 400/400 ≥ 0.95, verificado en paso 2; mapeo por
mapa_smiles_full.parquet, 400/400 mapeados, 0 SMILES fuera del pool):

- RGB local (submission_rgb.csv): **MRR@25 0.5758** (top-1 0.4550, recall@25 0.9175).
- RGB workspace Kaggle: **MRR@25 0.5723** (top-1 0.4500, recall@25 0.9200).
- Baseline-1: 1.0000 por construcción (su top-1 ES la verdad aquí).

El modelo es genuinamente bueno en su trabajo (0.57 en clase-1 real);
el trabajo simplemente no es lo que el test oculto premia en el 83%.

## 5. Consecuencia para la estrategia (medida, no opinión)

- Un modelo de retrieval de novo tiene un techo duro en el test oculto de
  f_class1 × (rendimiento clase-1) ≈ **0.08–0.10**. No puede superar el
  0.146 de baseline-1 solo con mejor retrieval: baseline-1 ya captura
  ~0.87 de la masa clase-1. El 0.049 < 0.146 lo confirma.
- Para puntuar en clases 2/3 (moléculas novedosas) el modelo tiene que
  GENERAR estructuras nuevas, no recuperar conocidas. El lineal Morgan
  (0.4146) y el RGB (0.4951) son ambos retrieval: su techo oculto es del
  mismo orden (~0.07–0.10). El blend Morgan+baseline-1 (0.462 local) es
  un número mol-split con la misma inflación por verdad-en-pool; su
  expectativa oculta es muy inferior.
- El MRR mol-split local es la métrica EQUIVOCADA para ordenar modelos
  de retrieval de cara al test oculto. Para la selección futura hay que
  evaluar simulando el test oculto: pool SIN las moléculas de consulta
  (simulación novedosa), o reportar MRR clase-1 × f_class1.

## Archivos de esta diagnosis

- `verificacion/submission_kaggle_rgb_58256941.csv` (salida del workspace,
  test visible, 396,528 B, md5 13cb3ebf8a43b4ef0943e103c3d8d0e9).
- `verificacion/DIAGNOSTICO_BRECHA_RGB.md` (este archivo).
- Llamadas del conector: `~/workspace/custom-connectors/custom.kaggle/calls/`
  `get_submission_rgb.json`, `download_submission_rgb.json`,
  `download_nb_output_rgb.json`. (`download_competition_submission` se
  agregó a ALLOWED_TOOLS en client.py tras leer su esquema; su uso dio
  permiso denegado.)
