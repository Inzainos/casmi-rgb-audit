# AGENTS.md — Enveda CASMI 2026

Reglas operativas para cualquier agente que trabaje en este repo.
Proyecto: Kaggle "Enveda CASMI 2026 — Molecule ID From Mass Spectra"
(competición de código, métrica MRR@25, envíos solo por notebook, sin
internet). Chat experto: "Qué es la entropía" (61e8d7ad). Coordinador:
chat "Coordinador" (aa46da50). Tablero: `~/workspace/coordinacion/TABLERO.md`.

## Reglas que no se rompen

1. **NUNCA ASUMAS, SIEMPRE VERIFICA.** Antes de reportar un resultado,
   léelo del archivo o del log. El estado "running/active" de un agente
   no es evidencia: se revisan los LOGS.
2. **Nada a Kaggle sin decisión de Elán.** Ningún envío, ningún cambio de
   selección final (2/2: generativa v2 (0.138) + envío 3 (0.146), cambiada
   por Elán 2026-11-23 21:53; la RGB (0.049) está fuera), ningún
   notebook/dataset público. Los envíos se hacen por navegador, con
   internet desactivado.
3. **Nada a GitHub.** Este repo no tiene remote configurado; Elán decide
   si se crea. Trabajo solo en local.
4. **Respaldos a Google Drive**, carpeta "Respaldos Fractal Core /
   Enveda CASMI 2026" (regla de Elán 2026-11-09 ~11:46). Nada de
   respaldos solo en local. GitHub = código versionado; Drive = respaldos.
5. **Código completo y con logs de ejecución.** Nada de fragmentos con
   "insertar resto aquí". Cada corrida deja su log en `verificacion/`.
6. **Los submissions existentes no se tocan.** `salida/submission.csv`
   (md5 7a7da389…), `salida/submission_paso2.csv` (md5 ed758b99…),
   `salida/submission_relleno25.csv` (md5 7f082916…),
   `salida/submission_morgan.csv` (md5 85458472…) se verifican por md5
   antes y después de cada paso.
7. **Fechas del concurso (verificadas en Kaggle 2026-11-09 ~20:54):**
   entrada y fusión de equipos 7 dic 2026 11:59 PM UTC; envío final
   14 dic 2026 11:59 PM UTC.

## Entorno (esta caja)

- Python 3.12.3 del sistema: numpy 1.26.4, pandas 2.1.4, scipy 1.11.4.
  **Nunca actualizar numpy a 2.x** en el python del sistema (pyarrow 17
  y scipy 1.11.4 necesitan numpy 1.x).
- **La caja se resetea y los paquetes de pip desaparecen.** Tras un
  reset, reinstalar lo que falte con `pip --break-system-packages`:
  pyarrow==17.0.0, scikit-learn 1.6.1, rdkit==2024.03.6 (el paquete PyPI
  se llama `rdkit`, no `rdkit-pypi`). El diagnóstico con RDKit 2026.03.3
  (la versión pinned del scorer) vive en un venv aparte
  (`~/workspace/casmi-diag-venv`, numpy 2.5.4) y no toca el sistema.
- ~7 GB RAM, 4 CPU (1.5 compartidas efectivas). **Un solo proceso pesado
  a la vez**; los runs largos se matan por OOM. La build de la librería
  usa RSS máx ~2.9 GB.
- Sin config global de git. Identidad local en los commits de Elán:
  deamon / deamon@local.

## Estructura

- `01_construir.py`, `casmi_lib.py` — librería bineada (0.01 Da,
  top-100 picos, precursor 100–2000 Da, ≥3 picos). Build full ~300 s.
- `02_validar.py` — validación spec-split / mol-split (semilla 42).
- `03_predecir_test.py` — búsqueda por precursor ±10 ppm, coseno,
  top-25 por InChIKey14.
- `04_huellas.py` … `09_predecir_test_paso2.py`, `paso2_lib.py`,
  `10_topo_baseline.py`, `11_topo_predecir.py`, `topo_lib.py` — paso 2
  (espectro → huella de Morgan 2048 bits).
- `11_rellenar_25.py` — relleno a 25 exactos repitiendo el top-1.
- `casmi_runtime_notebook.ipynb` — notebook que genera predicciones en
  runtime (el que se envió como envío 3).
- `salida/` — submissions locales (ver md5 en la regla 6).
- `verificacion/` — logs, JSONs de métricas, diagnósticos y evidencia
  (`runtime.json`, `diagnostico_scorer.md`, `envio3_evidencia.md`,
  `fechas_casmi_verificadas.md`, `metricas_paso2_*.json`).
- `datos/` — train.parquet (3 GB, SHA-256 9423f90e…), test.parquet
  (SHA-256 2fa29c4a…), sample_submission.csv. **Fuera de git**
  (.gitignore); son archivos oficiales de Kaggle, re-descargables.
- `intermedios/` — librería construida (1.2 GB, full_lib.npz etc.).
  **Fuera de git.**

## Estado verificado (2026-11-09 ~21:00)

- Baseline 1: spec-split MRR@25 0.8691, mol-split 0.0000 por
  construcción (solo resuelve la clase de novedad 1).
- Paso 2: kNN gana (MRR@25 0.0346 mol-split); topológico pierde
  (0.0148). Informe `kaggle-casmi-informe-paso2.md`.
- Envíos a Kaggle: 1 y 2 "Submission Scoring Error" (copia estática; el
  test visible se reemplaza por el oculto en el re-run). Envío 3
  (notebook en runtime, Version 3, scriptVersionId 367881767):
  **Succeeded, public MRR@25 = 0.146** (2026-11-09 20:04:39 CST).
  3 de 5 envíos diarios usados. 0/2 finales.
- Lectura honesta: el oculto trae clases 2 y 3 (f_class1 ≈ 0.17); el
  margen está en el camino de novo.
- Pendiente de Elán: blend baseline 1 + paso 2 kNN para el siguiente
  envío, o dejarlo así; dónde vive CASMI; selección final antes del
  14 dic.

## Estado verificado (2026-11-13 ~22:40)

- Finales 2/2: envíos 3 (0.146) + 4 (0.049), marcados 2026-11-12.
- Piloto B generativo: GEN-VAL MRR@25 0.3372, Tanimoto@1 0.6373,
  novedad 0.9651; umbrales (MRR > 0.01, Tanimoto > 0.267) pasan.
- Fix del beam aplicado y validado en Colab (beam 0.2525); zip v2 en
  Drive (md5 ca8ac5a8…).
- Notebook de envío generativo en validación local (run F, celda 6);
  dataset Kaggle en manifiesto (5 archivos, 95,557,499 B). Nada a
  Kaggle hasta verificar.
- Temp de Drive eliminado 2026-11-13 ~22:27 (orden de Elán).

## Estado verificado (2026-11-14 ~09:52)

- Envío generativo v1 (ref 58333078): público 0.004, no seleccionado; finales 2/2 intactas (3: 0.146 + 4: 0.049). Nada a Kaggle sin GO de Elán.
- Fix rueda RDKit completo (G-V1 PASA): contenedor Python 3.13, rueda cp313 (md5 a9699b01…); notebook v2 solo local (md5 48f14be3…); dataset v2 (6 archivos, 132,744,644 B) sin subir — en Kaggle sigue la v1 (95,557,499 B, 5 archivos).
- Rediseño v2 aplicado: G-V2 PASA (blindaje), 5a PASA (fallback; run 400 mol byte-idéntico a la salida visible v1, md5 ac9da78d…).
- G-V3: medición fiel al notebook (23,684 mol; qmol ≥0.95 = 0.1862, qspec = 0.5025) confirmada por Datos — sustituye a cobertura_val.json; criterio 3a retirado. Calibración espera a la etapa B (Colab, Elán).
- G4: etapa A completa (2,400 mol); etapa B en Colab por Elán; luego calibración, etapa C (G4.json), 5b, G-V6, G-V7.

## Estado verificado (2026-11-24 ~12:05)

- Finales 2/2: generativa v2 (0.138) + envío 3 (0.146), cambiadas por Elán 2026-11-23 21:53; la RGB (0.049) está fuera. (Antes: 3 + 4, marcados 2026-11-12.)
- Envío generativo v3 (ref elnzainoscorona/casmi-notebook-generativo-v3, Version 2): público 0.138 (2026-11-23 ~22:19), igual que la v2; no marcado como final. En el visible, 0/400 moléculas necesitaron generación (retrieval puro); el decoder v3 cargó pero nunca disparó. Dataset Versión 3 en Kaggle (casmi-generativo-pilotob-v1, 6 archivos, modelo.pt cfe4f369…).
- Tests B/C en la Kali: AMBOS PASAN (2026-11-21). Test B (beam forzado, 10 mol): validez 245/250 = 0.9800. Test C (blindaje sin RDKit, 400 mol): PASA funcional (4 filas con reorden de empates, P-C-1; librería Δ=256 nnz, P-C-2).
- Diagnóstico (2026-11-24 11:52, para Elán): el cuello es COBERTURA, no orden — 975/1,000 novelas sin verdad en ninguna lista; la rama generativa aporta 1 molécula de 1,200 (ZHHFMKYOKONAFI). Harness NO PASA (novel 0.0115 < 0.021).
- v4 en construcción (GO de Elán 2026-11-24 11:57): notebook v3 con la fusión V1 (beam inyectado en la rama retrieval, k=15) — la única fusión medida que gana sin perder (novel 0.0120860, clase-1 Δ=0, híbrido 0.1577244). La v3 se construyó con V0; V1 nunca se construyó en notebook. Checkpoint v3, portón 0.6.
- Limpieza completa (2026-11-21): 1,128 archivos, ~13.1 GiB, todo con respaldo verificado en Drive. train.parquet locales borrados (orden de Elán 21:32); la copia de la Kali y la de browser_downloads están intactas. Disco: 118 GB libres. datos/, intermedios/, staging vacíos.
- Pendientes sin stopper: subida de train.parquet a Drive (hasta el 30, Drive en cero; ruta web/resumable); datasets-nuevos-dominios (decisión de Elán); tarea Colab en pausa (ya no hace falta, B/C pasaron en la Kali).
