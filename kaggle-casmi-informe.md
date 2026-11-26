# Enveda CASMI 2026 — Baseline 1: búsqueda en librería espectral

**Competencia:** Enveda CASMI 2026 — identificar la estructura 2D de una molécula
desde sus espectros MS/MS. Métrica **MRR@25** (match por conectividad atómica:
InChIKey14 tras canonicalización tautomérica con RDKit). Entrada hasta
7 dic 2026, cierre 14 dic 2026. **Nada enviado a Kaggle; todo local.**

**Fecha:** 2026-11-09. **Trabajo:** `~/workspace/snt-taxonomia/kaggle-casmi/`.

## 1. Datos (verificados, no re-descargados)

| Archivo | Filas | SHA-256 |
|---|---|---|
| `datos/train.parquet` | 2,539,608 × 18 cols | `9423f90e…` |
| `datos/test.parquet` | 1,213 × 12 cols (400 moléculas) | `2fa29c4a…` |
| `datos/sample_submission.csv` | 400 molecule_id, placeholder CCO | — |

- Train: **275,810 moléculas únicas (inchikey14)**; espectros por molécula:
  mediana 6, media 9.21, p90 14, max 1,468; 17,033 moléculas con 1 espectro.
- Test: 400 moléculas, 1,213 espectros (media 3.03, mediana 3, max 9), todos
  **timsTOF**, precursor_mz 245–460 Da, energías de colisión 20/40/60 eV.
  Aductos test: `[M+H]+` 959, `[M-H]-` 193, `[M+CH2O2-H]-` 31, `[M+Na]+` 22,
  `[M+NH4]+` 4, `[M+K]+` 2, `[M+Cl]-` 2.
- Librerías en train (ingest_lib): enveda-180 1,153,785 (45.5%), pluskal_ms2
  527,581, riken 347,171, gnps 220,849, massbank 101,727, mona 92,416,
  spectraverse 50,933, msdial 40,765, drug_plus 2,545, enveda-np-examples 1,184,
  masaryk 652.
- Picos por espectro (train): media 158.2, mediana 42, p99 1,347, max 73,320.
  **Total 401.8M de picos** — por eso el pipeline es por chunks y con tope
  de picos (ver §2). Test: mediana 230 picos/espectro, media 330, max 3,259.
- `precursor_error_ppm` (train): mediana 1.51, p90 5.0, p99 2,884;
  |error| ≤ 5 ppm: 90.0%; ≤ 10 ppm: 93.1%; ≤ 20 ppm: 95.3%; ≤ 50 ppm: 96.0%.
  La cola pesada (>500 ppm: 3.4%) son anotaciones malas, no física.
- SMILES distintos por inchikey14: 1 en el 99.4% de las moléculas (2 en 0.5%,
  max 5). Aductos distintos por inchikey14: 1 en 45.1%, 2 en 33.0%, ≥3 en 21.9%.
- Cobertura de precursor (test vs train completo): 0/1,213 espectros sin
  candidatos, tanto a ±0.01 Da (mediana 1,088 candidatos) como a ±10 ppm
  (mediana 445).

## 2. Método exacto

**Bineado.** Bins de **0.01 Da** de 0 a 2,000 Da (200,000 columnas). Por espectro
se toman los **top-100 picos por intensidad** (las intensidades están
normalizadas con pico base = 1.0); picos del mismo bin se suman. Justificación:
los m/z tienen 4 decimales (alta resolución) y el test es timsTOF; 0.01 Da es
el compromiso entre discriminar picos cercanos y tolerar diferencias de
calibración entre instrumentos. La variante 0.05 Da se probó y perdió
(ver §3.3): se queda 0.01 Da.

**Filtros.** precursor_mz en [100, 2,000] Da, num_peaks ≥ 3, intensidad > 0.
Descarta 101,544 filas (4.0%: 3,119 por precursor, 98,425 por nº de picos).
**507 moléculas quedan sin ningún espectro válido** y no pueden ser candidatas.

**Representación.** Una CSR por librería con las intensidades crudas bineadas
(float32). El coseno y la entropía se derivan de la misma matriz.
- Librería de validación: 2,285,808 filas, 117,536,497 nnz (470 MB).
- Librería completa: 2,438,064 filas, 125,647,921 nnz (503 MB).
- Intermedios en `intermedios/` (1.2 GB con metas y mapas).

**Recuperación de candidatos.** Librería ordenada por precursor_mz;
por espectro de consulta se toman los espectros con precursor dentro de la
ventana (**±10 ppm con piso de 0.002 Da** — ver justificación en §3.2).
La variante ±0.01 Da se probó y perdió ligeramente.

**Similitudes.**
- **Coseno (principal):** intensidades con raíz cuadrada y normalización L2
  (estándar GNPS/matchms): cos = (√a·√b)/(‖√a‖‖√b‖).
- **Entropía espectral (variante, Li et al. Anal. Chem. 2021):** con
  p = intensidades normalizadas L1 y S = −Σ p ln p,
  sim = 1 − (2·S_AB − S_A − S_B)/ln 4, donde S_AB es la entropía del espectro
  fusionado. Se calcula solo con el término de intersección
  (S_AB = ln 2 + (S_A+S_B)/2 − ½·Σ_bins_comunes [g(a+b)−g(a)−g(b)],
  g(x) = x ln x). Self-check verificado: idéntico → 1.000000,
  disjunto → 0.000000 (coseno idéntico → 1.000000).

**Agregación por molécula.** Para cada molécula de consulta se combinan sus
espectros tomando, por cada inchikey14 candidato, la **máxima** similitud
entre sus espectros; se ordena y se devuelven **hasta 25 inchikey14 distintos**
(uno por inchikey14). El SMILES emitido por inchikey14 es el
**normalized_smiles más frecuente** en train para ese inchikey14.

**Métrica.** MRR@25 con match exacto por inchikey14 (en train el inchikey14
ya es la clave canónica). Rango 1-based dentro del top-25; 0 si no aparece.

## 3. Validación

### 3.1 Split (semilla 42)

- **Mol-split (principal, anti-fuga):** 5% de las moléculas reservadas:
  **13,790 moléculas (126,953 espectros)**; la librería son los espectros
  restantes (2,285,808 filas). 13,748 moléculas produjeron consultas
  (38,377 espectros; 42 moléculas perdieron todos sus espectros por el
  submuestreo hash y no aparecen ni en librería ni en consultas).
- **Spec-split (diagnóstico clase-1):** 5% de las moléculas con ≥4 espectros:
  **9,943 moléculas**; se retuvieron ~3 espectros por molécula como consultas
  (29,787 espectros, 9,936 moléculas con consultas) y el resto quedó en la
  librería. La selección garantiza ≥1 espectro en librería por molécula
  (máscara de holdout por hash; la primera versión con umbral probabilístico
  dejaba moléculas sin ningún espectro en librería y se corrigió).
- Integridad verificada por asserts: 0 moléculas de validación en la librería
  (mol-split); todas las moléculas spec-split están en la librería.

### 3.2 Resultados (bins 0.01 Da)

| Validación | Ventana | Métrica | MRR@25 | Top-1 | Top-5 | Recall@25 | Mediana rango |
|---|---|---|---|---|---|---|---|
| **mol-split** (n=13,748) | ±10 ppm | coseno | **0.0000** | 0.000 | 0.000 | 0.000 | — |
| **mol-split** (n=13,748) | ±10 ppm | entropía | **0.0000** | 0.000 | 0.000 | 0.000 | — |
| **mol-split** (n=13,748) | ±0.01 Da | coseno | **0.0000** | 0.000 | 0.000 | 0.000 | — |
| **mol-split** (n=13,748) | ±0.01 Da | entropía | **0.0000** | 0.000 | 0.000 | 0.000 | — |
| **spec-split** (n=9,936) | **±10 ppm** | **coseno** | **0.8691** | **0.801** | **0.952** | **0.987** | **1** |
| **spec-split** (n=9,936) | ±10 ppm | entropía | 0.8606 | 0.788 | 0.949 | 0.986 | 1 |
| **spec-split** (n=9,936) | ±0.01 Da | coseno | 0.8653 | 0.797 | 0.949 | 0.986 | 1 |
| **spec-split** (n=9,936) | ±0.01 Da | entropía | 0.8566 | 0.784 | 0.945 | 0.985 | 1 |

**Lectura honesta del mol-split = 0.000:** no es un fallo del método, es lo
que la métrica mide por construcción — la molécula de consulta NO está en la
librería, así que su inchikey14 no puede aparecer en ningún ranking. La
búsqueda en librería solo resuelve moléculas que ya están en la librería
(clase 1 de novedad). El número útil para el envío es el spec-split:
**MRR@25 ≈ 0.87 en moléculas que sí están en la librería**. El MRR@25
esperado en el test real es ≈ f_clase1 × 0.87, donde f_clase1 es la fracción
desconocida de las 400 moléculas de test que están en las librerías públicas
(las clases 2 y 3 puntúan ~0 con este método).

**MRR@25 por ingest_lib (spec-split, coseno, ±10 ppm):** pluskal_ms2 0.9661
(n=1,935), massbank 0.9518 (144), riken 0.9499 (143), mona 0.9278 (92),
gnps 0.8532 (515), **enveda-180 0.8413 (6,935)**, spectraverse 0.7613 (151).
(enveda-180, la librería más parecida al test, es la más difícil: más
isómeros/análogos cercanos compitiendo; spectraverse tiene n pequeña.)

**Justificación de la ventana ±10 ppm (con piso 0.002 Da):** el error de
precursor en train tiene mediana 1.51 ppm y p90 5.0 ppm, así que la diferencia
precursor consulta–librería para la misma molécula+aducto se concentra
dentro de ±10 ppm; el piso evita que la ventana sea absurdamente estrecha a
masas bajas (a 157 Da, 10 ppm = 0.0016 Da). La variante ±0.01 Da se probó en
la misma validación: MRR 0.8653 vs 0.8691 (coseno) con ~2.1× más candidatos
por espectro (mediana 791 vs 372), así que ±10 ppm gana en MRR y en coste.

### 3.3 Sensibilidad al ancho de bin (subconjunto 2,000 moléculas spec-split)

- 0.01 Da: coseno MRR@25 **0.8690**, entropía 0.8623.
- 0.05 Da (bins fusionados //5, consultas y librería alineadas):
  coseno 0.8641, entropía 0.8572.
- Se queda **0.01 Da** (gana por ~0.005 de MRR en ambas métricas).
- Nota: la primera corrida a 0.05 Da dio MRR 0.110 por un bug mío (fusioné
  los bins solo en la librería y no en las consultas: espacios desalineados).
  Se detectó por lo absurdo del número, se corrigió fusionando también las
  consultas y se repitió: 0.8641. El bug y la corrección quedan registrados;
  los números de la tabla de §3.2 no se vieron afectados (usan 0.01 Da).

## 4. Tiempos y memoria (medidos)

Máquina: 4 CPU, límite de cgroup ~6 GB. Python del sistema (pandas 2.1.4,
numpy 1.26.4, scipy 1.11.4, pyarrow 17.0.0; **scikit-learn NO está instalado
ni en el sistema ni en `~/workspace/py313-venv`** — irrelevante para este
baseline, se necesitará para el modelo de huellas).

| Paso | Tiempo | Pico RSS |
|---|---|---|
| Exploración (2 pasadas) | ~30 s | 506 MB |
| Split (semilla 42) | 96 s | 307 MB |
| Construcción librería validación | 340 s | 2,793 MB |
| Construcción librería completa | 352 s | 2,862 MB |
| Validación completa (4 configs) | 300 s | 2,524 MB |
| Predicción test | 18 s | 2,658 MB |

Incidentes: la primera construcción murió por OOM a los ~178 s (listas de
Python con ~137M de enteros: ~12 GB). Se reescribió con arrays numpy
prealocados a partir de una pasada escalar que acota el nnz por destino
(cota = Σ mín(num_peaks, 100)); pico resultante < 3 GB.

## 5. Envío (`salida/submission.csv`, NO enviado a Kaggle)

- Configuración: **coseno, ventana ±10 ppm (piso 0.002 Da), bins 0.01 Da**,
  librería completa (2,438,064 espectros).
- Formato verificado: **400 filas**, `molecule_id` en el mismo orden que
  `sample_submission.csv`, columna `smiles` con hasta 25 SMILES separados
  por `;`. SMILES por molécula: media 24.1, mediana 25, mín 2, máx 25;
  367 moléculas con 25; 0 vacíos, 0 nulos, 0 filas con un solo SMILES,
  todos los SMILES distintos dentro de cada fila, sin uso del fallback "CCO".
  Candidatos por molécula: mín 15, mediana 1,222, máx 11,992; 0 moléculas
  sin candidatos.
- md5: `7a7da389417fb162039a94394d76e1ee` (385,992 bytes).

## 6. Limitaciones honestas (qué NO mide este baseline)

1. **Solo resuelve clase 1.** Moléculas que no están en ninguna librería
   pública (clases 2 y 3) puntúan 0 con este método por construcción
   (mol-split MRR = 0.000). El MRR real del envío depende de la mezcla de
   clases del test, que no se conoce.
2. **Sin matches entre aductos.** La ventana es sobre precursor_mz, así que
   un espectro de consulta solo encuentra librería con el mismo aducto; el
   54.9% de las moléculas de train tienen ≥2 aductos. La vía es la
   recuperación por masa neutra (restar la masa del aducto).
3. **Tautómeros/estereoquímica:** el match es por inchikey14 (conectividad);
   el calificador canonicaliza tautómeros con RDKit, así que cualquier
   variante con la misma conectividad puntúa. El SMILES emitido es una
   variante representativa; no se distingue entre estereoisómeros.
4. **Sin energía de colisión ni instrumento:** los espectros se juntan sin
   distinguir 20/40/60 eV ni timsTOF/Orbitrap/QTOF. El test es todo timsTOF
   a 20/40/60 eV; la librería mezcla todo.
5. **Tope de 100 picos y bins de 0.01 Da:** los picos de baja intensidad se
   descartan y los picos que caen justo en un borde de bin se separan
   (la variante 0.05 Da pierde solo ~0.005 de MRR, así que el efecto es
   pequeño pero real).
6. **507 moléculas de train no tienen ningún espectro válido** (filtros) y
   no pueden ser candidatas nunca.
7. **El spec-split es optimista en un sentido:** las consultas de validación
   vienen de las mismas librerías/instrumentos que la librería; el test real
   es todo timsTOF de productos naturales (el proxy más cercano es
   enveda-180: MRR 0.8413, no 0.8691).
8. La entropía espectral está implementada y verificada (self-check 1.0/0.0),
   pero aquí rinde ligeramente peor que el coseno (0.8606 vs 0.8691); queda
   como variante reportada, no usada en el envío.

## 7. Siguiente paso recomendado

**Modelo espectro → huella de Morgan** (el salto que resuelve clases 2 y 3):
entrenar un modelo que prediga la huella de Morgan (2048 bits, radio 2) de
la molécula a partir del espectro bineado (o de pares pico-intensidad), y
luego recuperar candidatos de PubChem/COCONUT por similitud de Tanimoto
contra la huella predicha, re-ordenando con la búsqueda en librería donde
haya match. En paralelo, y más barato: (a) recuperación por masa neutra
(para cazar los matches entre aductos que este baseline pierde) y
(b) ensamblar coseno + entropía por rango. Para el modelo harán falta
RDKit (instalar) y scikit-learn (instalar: no está en el sistema) o
PyTorch; la librería bineada de `intermedios/` ya sirve como entrada.

## 8. Archivos

- `01_construir.py` — split + construcción de librerías (CSR).
- `02_validar.py` — validación MRR@25 (mol-split + spec-split).
- `03_predecir_test.py` — predicción del test y `salida/submission.csv`.
- `casmi_lib.py` — carga, coseno, entropía, agregación, MRR.
- `explorar.py`, `explorar2.py` — exploración de los datos.
- `intermedios/` — `val_lib.npz` (+meta), `val_qmol.npz`, `val_qspec.npz`,
  `full_lib.npz` (+meta), `mapa_smiles_{val,full}.parquet`, `mol_index.npy`.
- `verificacion/` — `split_resumen.json`, `build_stats_{val,full}.json`,
  `validacion_bm1.json`, `ranks_{qmol,qspec}_{ppm,da}_bm1.csv`,
  `test_detalle.csv`, logs de cada paso.
- `salida/submission.csv` — 400 filas, md5 `7a7da389417fb162039a94394d76e1ee`.
