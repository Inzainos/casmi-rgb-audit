# CASMI — Notebook en runtime (Baseline 1 que genera predicciones)

Fecha: 2026-11-09/10. Notebook: `casmi_runtime_notebook.ipynb` (7 celdas, 5 de código).
Constructor: `verificacion/construir_notebook_runtime.py`. Ejecutor local:
`verificacion/ejecutar_notebook_runtime.py`. Verificación independiente:
`verificacion/verificar_runtime.py` → `verificacion/runtime.json`.

## Por qué existe

Los 2 envíos previos de Baseline 1 dieron "Submission Scoring Error" sin score
(scriptVersionId 367850937 y 367876111). Diagnóstico cerrado
(`verificacion/diagnostico_scorer.md`): la química está bien (0 fallos de
10,000 bajo RDKit 2026.03.3, la versión pinned del scorer) y las filas cortas
no eran la causa. La causa real es documental (Data page del concurso):
**el test visible son ejemplos del train y se reemplaza por el test oculto
durante el re-run**. Un notebook que solo copia un archivo estático produce
los `molecule_id` del test público, que no existen en el oculto → error
genérico. Este notebook **genera las predicciones en runtime** desde el
`test.parquet` que reciba, así que funciona con el test oculto.

## Método (idéntico al Baseline 1, informe §2)

- Librería bineada desde `train.parquet`: bins 0.01 Da (0–2000 Da), top-100
  picos por espectro, precursor 100–2000 Da, ≥3 picos.
- Por molécula del test recibido: candidatos por precursor (±10 ppm, piso
  0.002 Da), similitud coseno (raíz cuadrada + L2, estándar GNPS), agregación
  por molécula con el máximo por InChIKey14, top-25 SMILES representativos
  en orden de similitud.
- Relleno a exactamente 25 por fila repitiendo el top-1 (mismo criterio que
  `salida/submission_relleno25.csv`); si una molécula no tiene ningún
  candidato, "CCO" x25.
- El código es el de `01_construir.py` (modo full) + `03_predecir_test.py` +
  `casmi_lib.py`, adaptado: rutas parametrizadas, construcción en memoria
  (sin round-trip a npz), sin dependencias de módulos locales.

**Diferencias deliberadas vs `03_predecir_test.py` (documentadas):**
1. En 03, las moléculas cuyos espectros fueron todos descartados no aparecían
   en la salida. Aquí TODOS los `molecule_id` del test recibido llevan fila
   (fallback "CCO" x25). En el test público no cambia nada (0 descartados).
2. La salida va en orden alfabético de `molecule_id` (determinista), no en
   el orden de `sample_submission.csv` (que en el re-run trae los ids
   públicos, no los ocultos). En el test público ambos órdenes coinciden.
3. La verificación final del notebook es genérica (vale para el test público
   y el oculto): no compara contra `sample_submission.csv`.

## Verificación local de punta a punta (2026-11-10 ~01:47)

Ejecución completa del notebook contra los datos locales, con la salida en
`verificacion/runtime_check/submission.csv` (los intermedios existentes no
se tocaron y los 3 submissions previos quedaron intactos).

| Medida | Valor |
|---|---|
| Celda 1 setup | 1.1 s |
| Celda 2 build librería | 297.5 s |
| Celda 3 orden + stats | 10.3 s |
| Celda 4 predicción | 4.2 s |
| Celda 5 verificación | 0.0 s |
| **Total** | **313.1 s (5.2 min)** |
| **RSS máx proceso** | **2,834 MB** |
| Librería construida | 2,438,064 filas, 125,647,921 nnz, rep 275,303 moléculas |
| Stats build | 2,539,608 leídas; 3,119 filtradas pmz; 98,425 filtradas npeak; 0 sin picos |
| Test bineado | 1,213 espectros, 99,187 nnz, 0 descartados |

**Reproducción del Baseline 1 (verificación independiente, `runtime.json`):**

| Check | Resultado |
|---|---|
| Filas | 400 |
| Candidatos por fila | exactamente 25 en las 400 (10,000 total) |
| md5 de la salida | `7f082916a4a2bd68392750ea3cf2c8a7` — **idéntico a `submission_relleno25.csv`** |
| Filas idénticas a relleno25 (por molecule_id) | 400/400, 0 diferentes |
| Filas con prefijo original intacto | 400/400 |
| SMILES válidos en RDKit (2024.03.6) | 10,000/10,000, 0 inválidos |
| Sin nulos / sin vacíos / ids únicos | OK |
| Veredicto | **LISTO** |

La reproducción es byte por byte: el mismo código, los mismos datos y
determinismo completo dan el mismo archivo que el relleno25. Los 3
submissions previos siguen intactos
(`submission.csv` 7a7da389…, `submission_relleno25.csv` 7f082916…,
`submission_paso2.csv` ed758b99…).

## Checklist de cumplimiento (verificado por grep sobre el notebook)

| Requisito | Estado |
|---|---|
| Sin internet (sin requests/urllib/http/socket/subprocess) | OK — 0 ocurrencias en las celdas de código; 2 menciones en el texto de documentación (verificado por Mole 2026-11-09) |
| Solo datos de la competencia (`train.parquet` + `test.parquet` del input) | OK — resueltos por glob bajo `/kaggle/input`, fallback local solo para desarrollo |
| Sin datos externos | OK |
| Sin RDKit en runtime | OK — 0 ocurrencias en las celdas de código; las menciones son texto de documentación; imports solo hashlib/numpy/pandas/pathlib/pyarrow/scipy |
| Sin módulos locales (autocontenido) | OK — no importa `casmi_lib` ni scripts (las 3 menciones son comentarios) |
| Runtime total < 9 h | OK — 313 s local |
| Memoria dentro de CPU Kaggle (~13 GB) | OK — RSS máx 2,834 MB |
| Formato de salida según reglas | OK — header `molecule_id,smiles`; una fila por `molecule_id` del test; ≤25 guesses; sin nulos ni repetidos |

**Advertencias honestas:**
- El tiempo y la memoria se midieron en esta máquina (4 CPU compartidas,
  7 GB RAM). En Kaggle la CPU puede ser distinta, pero el margen es amplio
  (5 min de 9 h; 2.9 GB de ~13 GB).
- El score real será menor que el 0.8691 del spec-split local: el test oculto
  incluye las clases de novedad 2 y 3, que la búsqueda en librería no
  resuelve. El notebook no cambia el método, solo lo hace ejecutable en el
  re-run.
- El notebook debe enviarse con **internet desactivado** (requisito del
  concurso; el primer intento de envío falló por eso).

## Qué cambia vs los envíos 1 y 2

| | Envío 1 (V3) | Envío 2 (V4) | Este notebook |
|---|---|---|---|
| Contenido | copia estática de `submission.csv` | copia estática de `submission_relleno25.csv` | genera predicciones en runtime |
| Funciona con test oculto | No | No | Sí (lee el test que reciba) |
| Resultado Kaggle | Scoring Error | Scoring Error | **Succeeded, public MRR@25 = 0.146** (envío 3, 2026-11-09 20:04:39 CST, scriptVersionId 367881767; evidencia en `verificacion/envio3_evidencia.md`) |

## Envío a Kaggle — EXITOSO (2026-11-09 20:04:39 CST)

Notebook `casmi-runtime-notebook` (privado, vinculado al concurso),
Version 3, scriptVersionId **367881767**, internet desactivado. Corrida
en Kaggle 431.8 s; salida submission.csv (400.18 kB) con md5
`7f3fafa4ae01da0524ef5b18d8844ae0` (distinto al local, confirma test
oculto). Status: **Succeeded**, public MRR@25 = **0.146**. Es el 3º de 5
envíos diarios (quedan 2). Nada marcado como final (0/2). Evidencia
archivada en `verificacion/envio3_evidencia.md` (verificación de solo
lectura con la sesión de Elán, 2026-11-09 ~20:55, sin cambios).
