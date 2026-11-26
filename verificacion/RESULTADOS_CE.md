# Resultados CE-como-feature — CASMI (Enveda CASMI 2026)

**Fecha:** 2026-11-11 (junta ~12:39, orden de Elán: "en lo que lo suben inicien con CE").
**Máquina:** sandbox Kali (`deamonx`), `/home/deamon/wt_deamonx/CASMI-sandbox/pruebas/`.
Venv `/home/deamon/.venv-casmi` (RDKit 2026.09.1), Python 3.14.7. Nada a Kaggle, nada a main.
**Scripts:** `25_preparar_ce.py`, `26_entrenar_lineal_ce.py` (copia de `06_entrenar_lineal.py` +
`--ce-mode none/first/mean/onehot --tag`), `15_evaluar_pred.py`, `casmi_lib.mrr_from_ranks`.
**Criterio por variante:** MRR@25 mol-split > 0.4146 (referencia lineal_bin).

## 0. Referencia reproducida antes de las variantes

Modelo lineal_bin de la fase Morgan, re-entrenado tal cual con `26_entrenar_lineal_ce.py
--tag lineal_ref --ce-mode none` (K=50,000, 261,536 filas train, 6 épocas, batch 1024,
lr 5.0, lam 1e-4, momentum 0.9, semilla 42):

| | MRR@25 | top-1 | top-5 | recall@25 | mediana rango | cand. mediana/media |
|---|---|---|---|---|---|---|
| lineal_ref **bin** (clip [0,1] + top-100 bits) | **0.4146** (0.41462) | 0.2910 | 0.5644 | 0.8011 | 2 | 106 / 141.0 |
| lineal_ref crudo | 0.3721 (0.37207) | 0.2500 | 0.5185 | 0.7700 | 3 | 106 / 141.0 |

MSE por época idéntico al `06` original hasta ~15 dígitos:
0.01655, 0.01588, 0.01585, 0.01584, 0.01584, 0.01584. Referencia **reproducida exactamente**,
pipeline de evaluación confirmado. n=13,748, filtro de masa ±10 ppm, piso 0.002 Da, split mol.

## 1. Tabla CE y verificación de alineamiento (previo al entrenamiento)

Tabla en `pruebas/intermedios/ce_val_lib.npz`, `ce_val_qmol.npz`, `ce_val_qspec.npz`,
`ce_full_lib.npz` (script `24_tabla_ce.py`, informe `pruebas/verificacion/tabla_ce.json`).
Contenido por fila: `ce_n`, `ce_first`, `ce_mean`, `ce_min`, `ce_max`.
Alineamiento verificado antes de entrenar: conteos exactos (2,285,808 / 38,377 / 29,787) y
`precursor_mz` + `mol` elemento a elemento, como en el build de la tabla — **cuadró exacto**
(max diff 0.0, 0 mol mismatches); la selección de `05` se reprodujo exactamente
(`train_mol coincide=True`).
Distribución val_lib: 1 valor 66.3%, multi 20.2%, sin CE 13.6%; mediana ce_first 22,
ce_mean 40; max 466.0 (outlier; >100 son 13,684 filas, 0.6%); 0 negativos.
Train sin CE 49,348 (18.9%), val sin CE 4,463 (11.6%).
One-hot train buckets [133116, 26353, 15595, 37124, 49348], val [11758, 7849, 7191, 7116, 4463].

## 2. Diseño de las variantes (decisiones fijadas)

- **V1 `ce_first`**: UNA sola columna añadida a las features binarias. Recorte a [0,100],
  dividir entre 100 (escala 0–1, igual que las binarias), sin CE → imputar mediana
  22/100 = **0.22**. Media medida de la columna CE en train: 0.2421.
- **V2 `ce_mean`**: mismo diseño con `ce_mean`. Sin CE → mediana 40/100 = **0.40**.
  Media medida de la columna CE en train: 0.3832.
- **V3 `ce_onehot`**: one-hot de 5 buckets {20, 40, 60, otro, sin CE} añadido a las
  features binarias (primera prueba barata del "modelo por CE"; los modelos separados por
  CE quedan como seguimiento si esta ganaba). Medias medidas de las 5 columnas en train:
  [0.5085, 0.1007, 0.0596, 0.1419, 0.1888] = exactamente las fracciones de los buckets
  train [133116, 26353, 15595, 37124, 49348] / 261,536.

Todo lo demás idéntico a la referencia: mismo split, mismo filtro, misma validación,
misma semilla, 6 épocas, batch 1024, lr 5.0, lam 1e-4, momentum 0.9.

## 3. Resultados (MRR@25 mol-split, n=13,748)

| Variante | MRR@25 bin | MRR@25 crudo | top-1 bin | top-5 bin | recall@25 bin | mediana rango bin | MSE final | \|W\|_F | Veredicto |
|---|---|---|---|---|---|---|---|---|---|
| lineal_ref (sin CE) | **0.4146** | 0.3721 | 0.2910 | 0.5644 | 0.8011 | 2 | 0.01584 | 122.2 | referencia |
| V1 ce_first | 0.3829 | 0.3387 | 0.2631 | 0.5246 | 0.7703 | 3 | 0.01574 | 110.8 | **NO GANA** |
| V2 ce_mean | 0.3726 | 0.3331 | 0.2549 | 0.5113 | 0.7633 | 3 | 0.01563 | 103.0 | **NO GANA** |
| V3 ce_onehot | 0.3517 | 0.3250 | 0.2356 | 0.4839 | 0.7460 | 3 | 0.01625 | 98.6 | **NO GANA** |

Deltas bin contra 0.4146: ce_first −0.0317 (−7.7%), ce_mean −0.0420 (−10.1%),
ce_onehot −0.0629 (−15.2%). Crudo: −0.0334, −0.0390, −0.0471 respectivamente.
Candidatos/mol mediana 106, media 141.0 en las cuatro corridas (mismo filtro, mismo split).

MSE por época:
- lineal_ref: 0.01655, 0.01588, 0.01585, 0.01584, 0.01584, 0.01584
- ce_first: 0.01639, 0.01579, 0.01575, 0.01574, 0.01574, 0.01574
- ce_mean: 0.01623, 0.01567, 0.01564, 0.01564, 0.01563, 0.01563
- ce_onehot: 0.01692, 0.01625, 0.01623, 0.01624, 0.01623, 0.01625

Tiempos (Kali, 1 proceso pesado a la vez): lineal_ref 3,431.8 s; ce_first 2,943.8 s;
ce_mean 3,167.4 s; ce_onehot 3,339.6 s. RSSmax ~2,654–2,660 MB en todos. Sin contención
observada (cada entrenamiento se lanzó solo después de verificar que el anterior había
terminado; nada más pesado corriendo en la Kali en esos momentos).

## 4. Lectura (medido, no supuesto)

- **Las tres variantes pierden.** Ninguna supera 0.4146 en bin ni en crudo.
- **El MSE de entrenamiento NO predice el MRR aquí.** `ce_first` y `ce_mean` bajan el MSE
  final (0.01574 y 0.01563 vs 0.01584 de la referencia) y sin embargo empeoran el MRR@25.
  `ce_onehot` ni siquiera baja el MSE (0.01625, peor que la referencia) y es la peor en MRR.
- **Señal de desplazamiento de la predicción:** la predicción media por espectro sube con CE
  (ref 0.0171, max 1.863 → ce_first 0.0215, max 2.361; ce_mean 0.0214, max 2.214;
  ce_onehot 0.0221, max 1.788). La columna CE es una señal escalar fuerte que mueve la
  predicción global sin mejorar el ranking por Tanimoto (Tanimoto continuo medio bin:
  ref 0.2421 → first 0.2359, mean 0.2340, onehot 0.2250).
- **Hipótesis (no medida):** dentro del filtro de masa (±10 ppm) las moléculas candidatas
  de una misma consulta tienden a compartir CE parecida, así que la CE aporta poco para
  discriminar entre candidatas y solo añade ruido de escala al ranking. NO se midió la
  correlación CE–rango ni la dispersión de CE dentro de cada lista de candidatos; queda
  como hipótesis, no como hecho.
- **Lo que NO se hizo:** modelos separados por CE (quedaban condicionados a que el one-hot
  ganara; no ganó). Tampoco se probó CE como feature de re-ranking (segunda etapa) en vez
  de dentro del lineal, ni otros recortes/escalas.

## 5. Veredicto general

**CE-como-feature dentro del modelo lineal NO mejora el MRR@25 mol-split.**
El mejor medido sigue siendo el lineal_bin sin CE (0.4146). La comparación final del punto 4
(RGB vs 0.146 / Morgan / blend) sigue con Morgan = 0.4146 como referencia de la fase Morgan;
la CE no cambia ese número.

## 6. Archivos

Kali, `pruebas/`:
- `RESULTADOS_CE.md` (este reporte)
- `25_preparar_ce.py`, `26_entrenar_lineal_ce.py`, `24_tabla_ce.py`, `15_evaluar_pred.py`
- `verificacion/entrenar_lineal_{lineal_ref,ce_first,ce_mean,ce_onehot}.json` (MSE por época, K, hiperparámetros)
- `verificacion/metricas_ce_{first,mean,onehot}_val_mol.json` (bin) y
  `verificacion/metricas_ce_{first,mean,onehot}_crudo_val_mol.json` (crudo)
- `verificacion/metricas_masa_lineal_ref_val_mol.json` y
  `verificacion/metricas_masa_lineal_ref_crudo_val_mol.json` (referencia reproducida)
- `verificacion/26_{lineal_ref,ce_first,ce_mean,ce_onehot}.log` (logs de entrenamiento)
- `verificacion/15_{lineal_ref,ce_first,ce_mean,ce_onehot}_{bin,crudo}.log` (logs de evaluación)
- `intermedios/P_val_{lineal_ref,ce_first,ce_mean,ce_onehot}.npy` (112,623,744 B c/u),
  `intermedios/W_lineal_{lineal_ref,ce_first,ce_mean,ce_onehot}.npy` (~410 MB c/u),
  `intermedios/P_val_mol_{...}.npy` (55,120 B c/u)

Copia en esta caja: `~/workspace/enveda-casmi-2026/verificacion/`
(`metricas_ce_*_val_mol.json`, `metricas_ce_*_crudo_val_mol.json`,
`metricas_masa_lineal_ref_*_val_mol.json`, `RESULTADOS_CE.md`), verificada por md5.

## 7. Incidencias

- La sesión SSH lanzadora del vigilante de `ce_onehot` murió por SIGTERM del runtime
  (~15:45 CST, reinicio del runtime). Sin impacto: el entrenamiento y el vigilante remoto
  corrían con nohup en la Kali y sobrevivieron; se relanzó un vigilante local rastreado
  que entregó el resultado completo a las 16:22 CST.
- Ningún OOM, ningún archivo faltante, ninguna falla de alineamiento. La referencia se
  reprodujo exactamente antes de correr las variantes, como pedía la tarea.
- Nada a Kaggle, nada a main, nada a GitHub. Solo trabajo en el sandbox de la Kali y
  copias de verificación en esta caja.
