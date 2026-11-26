# CASMI — Relleno a 25 candidatos del Baseline 1

**Fecha:** 2026-11-09. **Motivo:** el envío del Baseline 1 a Kaggle
(~17:17, notebook privado, scriptVersionId 367850937) devolvió
"Submission Scoring Error", sin score. Diagnóstico verificado: los 9,641
SMILES son válidos en RDKit, sin vacíos ni duplicados; la única desviación
es que 33 filas traen menos de 25 candidatos (367 con 25) y el scorer
exige 25 exactos aunque las reglas digan "hasta 25". Decisión de Elán:
rellenar las 33 filas cortas a 25 y reenviar.

## Método (script `11_rellenar_25.py`)

- Misma búsqueda que `03_predecir_test.py`: mismo bineado (top-100 picos,
  bins 0.01 Da), misma ventana de precursor (±10 ppm, piso 0.002 Da),
  mismo coseno (raíz cuadrada + L2), misma agregación por molécula
  (máxima similitud por inchikey14, `rank_molecule`).
- Diferencia única: NO se corta la lista rankeada en 25; se toman los
  candidatos rankeados después de los ya listados, en orden de similitud.
- Si la lista completa dedupada tiene <25 entradas: se completa
  repitiendo el candidato top-1 (mayor similitud). NO se usa "CCO" y NO se
  ensancha la ventana del precursor. Mismo Baseline 1, mismo protocolo,
  solo el relleno.
- Seguridad interna del script: para cada fila rellenada se verifica que
  la lista completa reproducida empiece EXACTAMENTE con los candidatos
  originales, en el mismo orden (misma búsqueda que 03). Las 33
  verificaciones pasaron.

## Resultado

- Archivo nuevo: `salida/submission_relleno25.csv`,
  **md5 `7f082916a4a2bd68392750ea3cf2c8a7`** (distinto del original).
- Original intacto: `salida/submission.csv`,
  md5 `7a7da389417fb162039a94394d76e1ee` verificado antes y después.
- **Hallazgo: NO hay más candidatos reales.** En las 33 filas, la lista
  completa dedupada tiene exactamente la misma longitud que la original
  (2–24 entradas): las ventanas de ±10 ppm de esas moléculas no contienen
  más moléculas distintas en la librería. Por eso el relleno es 100%
  repeticiones del top-1: **359 repeticiones en total, 0 candidatos
  reales añadidos.**

## Tabla de relleno por fila

| molecule_id | n_original | n_lista_completa | reales añadidos | repeticiones top-1 |
|---|---|---|---|---|
| m_08d7f1 | 5 | 5 | 0 | 20 |
| m_0cc9ea | 10 | 10 | 0 | 15 |
| m_1a0749 | 15 | 15 | 0 | 10 |
| m_1f86e6 | 14 | 14 | 0 | 11 |
| m_214040 | 5 | 5 | 0 | 20 |
| m_2363c0 | 19 | 19 | 0 | 6 |
| m_2c2340 | 16 | 16 | 0 | 9 |
| m_436c77 | 10 | 10 | 0 | 15 |
| m_47617a | 21 | 21 | 0 | 4 |
| m_568a04 | 13 | 13 | 0 | 12 |
| m_6b66aa | 17 | 17 | 0 | 8 |
| m_6f45e5 | 19 | 19 | 0 | 6 |
| m_76ffc5 | 10 | 10 | 0 | 15 |
| m_773a95 | 21 | 21 | 0 | 4 |
| m_7ee70f | 17 | 17 | 0 | 8 |
| m_847112 | 14 | 14 | 0 | 11 |
| m_8d75c7 | 11 | 11 | 0 | 14 |
| m_935b22 | 24 | 24 | 0 | 1 |
| m_997dbc | 5 | 5 | 0 | 20 |
| m_9f1fb6 | 6 | 6 | 0 | 19 |
| m_b53b6e | 2 | 2 | 0 | 23 |
| m_b7be8a | 16 | 16 | 0 | 9 |
| m_b9f2b4 | 7 | 7 | 0 | 18 |
| m_bb75e4 | 18 | 18 | 0 | 7 |
| m_bcd97d | 7 | 7 | 0 | 18 |
| m_c50497 | 22 | 22 | 0 | 3 |
| m_cd6458 | 12 | 12 | 0 | 13 |
| m_d274b6 | 24 | 24 | 0 | 1 |
| m_d7ad1b | 23 | 23 | 0 | 2 |
| m_e138be | 24 | 24 | 0 | 1 |
| m_e1d7ad | 14 | 14 | 0 | 11 |
| m_e5b5fa | 6 | 6 | 0 | 19 |
| m_ed201b | 19 | 19 | 0 | 6 |
| **Total** | — | — | **0** | **359** |

Detalle por fila en `verificacion/relleno25_detalle.csv`; números
consolidados en `verificacion/relleno25.json`; log en
`verificacion/relleno25.log`.

## Verificación independiente (desde los datos, no solo el log)

- 400 filas; orden de `molecule_id` idéntico a `sample_submission.csv`.
- EXACTAMENTE 25 candidatos en cada fila (10,000 en total).
- Todos los SMILES válidos en RDKit (0 inválidos), 0 candidatos vacíos.
- 367 filas byte-idénticas al original (mismos SMILES, mismo orden).
- 33 filas rellenadas conservan sus candidatos originales primero y en
  el mismo orden; lo añadido son solo repeticiones del top-1 (359).
- md5 nuevo `7f082916a4a2bd68392750ea3cf2c8a7`, distinto del original.

## Lectura honesta del relleno

El relleno es solo cumplimiento de formato, no cambia el MRR esperado:
las repeticiones van DESPUÉS de los candidatos originales, así que el
rango de la molécula verdadera (si está en la lista original) no se
mueve, y si no está, repetir el top-1 no la añade. El scorer
canoniza por InChIKey14, así que las repeticiones colapsan a la misma
clave que el top-1 ya listado. Además el propio `sample_submission.csv`
de la competencia trae 25 "CCO" idénticos por fila, así que los
candidatos duplicados son aceptados por el formato. La única duda que
queda es empírica: si el scorer acepta el archivo rellenado, lo dirá el
reenvío (2º de 5 envíos diarios). Nada enviado a Kaggle desde aquí.

## Archivos

- `11_rellenar_25.py` — script de relleno.
- `salida/submission_relleno25.csv` — archivo rellenado (md5 arriba).
- `verificacion/relleno25.json` — números consolidados.
- `verificacion/relleno25_detalle.csv` — tabla por fila.
- `verificacion/relleno25.log` — log de la corrida.
