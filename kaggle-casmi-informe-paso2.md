# Enveda CASMI 2026 — Paso 2: modelo espectro → huella de Morgan

**Competencia:** Enveda CASMI 2026 — identificar la estructura 2D de una molécula
desde sus espectros MS/MS. Métrica **MRR@25** (match por conectividad atómica:
InChIKey14 tras canonicalización tautomérica con RDKit). Entrada hasta
7 dic 2026, cierre 14 dic 2026. **Nada enviado a Kaggle; todo local.**

**Fecha:** 2026-11-09. **Trabajo:** `~/workspace/snt-taxonomia/kaggle-casmi/`.
**Base:** baseline 1 (`kaggle-casmi-informe.md`): búsqueda en librería espectral,
spec-split MRR@25 0.8691, mol-split MRR@25 0.000 por construcción. Este paso
ataca las clases 2 y 3 (moléculas nuevas) prediciendo la huella de Morgan
desde el espectro y recuperando candidatos del pool por Tanimoto.

**Cambio de base (orden de Elán, 2026-11-09 ~09:47):** el paso 2 NO se
construye desde cero con un regresor genérico. El núcleo es el **agente
topológico**: el SNT Genomic Topologic Analyzer (repo público
Inzainos/Genomic_agent, MIT, HEAD 28450d5; clon local
`~/workspace/genomic-agent/`), adaptado a CASMI. El modelo lineal queda
como **control** (§4) y el kNN como segunda comparación (§5). Se reporta
lo que miden los números, sin forzar que la topología gane.

## 1. Entorno (instalado y verificado)

Python 3.12.3 del sistema. **CUIDADO con numpy:** pyarrow 17.0.0 y scipy
1.11.4 exigen numpy 1.x, así que numpy se quedó en **1.26.4** y se fijaron
versiones compatibles:

| Paquete | Versión | Nota |
|---|---|---|
| numpy | 1.26.4 | fijado, no se tocó |
| scipy | 1.11.4 | fijado, no se tocó |
| pyarrow | 17.0.0 | **no estaba instalado** (el baseline 1 lo usaba; se reinstaló) |
| scikit-learn | 1.6.1 | `scikit-learn==1.5.3` no existe (la serie salta de 1.5.2 a 1.6.0); 1.6.1 es compatible con numpy 1.26 |
| RDKit | 2024.03.6 | `rdkit-pypi`; la serie 2024.09 exige numpy 2, por eso se fijó 2024.03.6 |
| pandas | 2.1.4 | del sistema |

Verificación: los 5 importan; RDKit genera la huella de Morgan de CCO
(6 bits) y `GetMorganGenerator(radius=2, fpSize=2048)` sin el warning de
deprecación de `GetMorganFingerprintAsBitVect`.

**Nota de entorno 2026-11-09 ~17:00 (reinstalación):** al correr el
script de test (09), `pyarrow` ya no estaba instalado — el entorno se
había reiniciado y `/usr/local/lib/python3.12/dist-packages/` estaba
vacío (sklearn, RDKit y pyarrow perdidos; solo quedaban los paquetes
del sistema en `/usr/lib/python3/dist-packages`). Se reinstalaron las
mismas versiones con `pip3 install --break-system-packages`
(PEP 668 bloquea `pip install` sin ese flag en este Python):
`numpy==1.26.4`, `pyarrow==17.0.0`, `scikit-learn==1.6.1` y
**`rdkit==2024.03.6`**. Detalle: el paquete de PyPI ya NO se llama
`rdkit-pypi` (ese nombre no existe: "from versions: none") — ahora es
**`rdkit`** (el dry-run ofrece `rdkit-2026.9.1`, que exigiría numpy 2,
por eso se fijó `2024.03.6`). Verificado tras la reinstalación: numpy
1.26.4, scipy 1.11.4, pyarrow 17.0.0, sklearn 1.6.1, rdkit 2024.03.6,
pandas 2.1.4; Morgan de CCO = 6 bits.

Máquina: 4 CPU, límite de cgroup ~6 GB. Picos de RSS medidos por paso
(ver §6).

## 2. Huellas del pool (04_huellas.py)

- Fuente: `mapa_smiles_full.parquet` (SMILES representativo = el más
  frecuente por inchikey14): **275,303 moléculas** con SMILES. Las 507
  restantes no tienen ningún espectro válido y no pueden ser candidatas.
- Huella: Morgan, **2048 bits, radio 2** (`rdFingerprintGenerator`).
- Resultado: **275,303 con huella, 0 fallos de parseo**, 12,678,178 bits
  en total. Bits por molécula: media 46.1, mediana 46, p10 36, p90 55,
  mín 3, máx 203. Los 2048 bits se usan (0 sin uso); 873 bits con
  frecuencia ≥1%. Frecuencia media de bit 0.0224.
- Salida: `intermedios/huellas_pool.npz` (CSR bool 275,810×2048; la fila i
  es el código de molécula i; filas vacías = sin huella) y
  `huellas_freq.npz` (frecuencias por bit, para el baseline nulo).
- Tiempo: 119 s, RSS máx 586 MB.

## 3. Modelo central — topológico (adaptación del Genomic Topologic Analyzer)

### 3.1 El agente original (leído antes de adaptar)

Fuentes leídas: `agent_core/agent_logic.py` (1,196 líneas),
`README.md` y la versión matemática limpia
`~/workspace/snt-taxonomia/kaggle-genomico/genomico-agent-core.py`
(524 líneas). El motor, en sus propios términos:

- **Red de referencia (baseline):** 51 pares hub-satélite, 14 hubs,
  11 cromosomas, en SQLite con `mean_ratio`/`std_dev_ratio` por par
  (50/51 derivados de n=40 muestras TCGA-BRCA sanas; solo NRAS→PI3K
  sintético). **Oráculo de enfermedad:** 44 filas, 17 enfermedades.
- **Nivel 1 — triage O(K):** sobre las firmas del oráculo,
  `R = TPM(satélite)/TPM(hub)`, `Z = (R_paciente − μ_sano)/σ_sano`,
  `|Z| > 2.5` → confirmado. Se omite la firma si el satélite falta del
  panel (faltante ≠ cero medido) o el hub no se expresa (TPM ≤ 0).
  Si σ = 0, Z = 0.
- **Nivel 2 — escáner por bloques:** recorre TODOS los pares por
  cromosoma (un bloque en RAM a la vez, se libera al terminar),
  mismos Z; los pares con |Z| > 2.5 no cubiertos por el Nivel 1 son
  **anomalías huérfanas** (el camino de novo).
- **Tipos de anomalía:** LEAPFROG (Z >> +2.5, el satélite se desacopla
  hacia arriba), SATELLITE_CAPTURE (Z << −2.5, gen libre arrastrado al
  control del hub), HUB_COLLAPSE (el hub pierde el control global de
  su regulón; tipo de oráculo, Nivel 1).
- Validación del agente: TCGA-BH-A18H Nivel 1 10/25 confirmados con Z
  exactos; Nivel 2 13 huérfanas (no 14: la 14ª era un artefacto de
  contar un gen ausente como TPM 0, corregido en el repo, commit
  28450d5).

### 3.2 Mapeo a CASMI (propuesta validada con los datos, no dada por buena)

| Genómico | CASMI |
|---|---|
| Paciente (perfil de expresión) | Molécula (sus espectros MS/MS) |
| Hub (gen regulador) | La molécula misma |
| Satélite (gen) | Bit de huella de Morgan (subestructura) |
| TPM(sat) / TPM(hub) | R_b = evidencia satélite / evidencia del hub (ver §3.3) |
| Baseline sano (n=40) | Librería de train (1 espectro por molécula, 261,536) |
| Oráculo (44 firmas) | Bits bien caracterizados = Nivel 1 (1,845 bits, §3.4) |
| Cromosomas (bloques) | Bloques de 256 bits (acotan la RAM) |
| Z = (R−μ)/σ, \|Z\| > 2.5 | Igual umbral, igual fórmula |
| LEAPFROG (Z >> +2.5) | Bit PRESENTE (evidencia satélite anómalamente alta) |
| SATELLITE_CAPTURE (Z << −2.5) | Bit AUSENTE con confianza |
| HUB_COLLAPSE | Molécula sin ningún bit presente (espectro degenerado) |

### 3.3 El observable R_b y el denominador del hub

**Primera versión (descartada por diagnóstico):** R_b = |S_b ∩ Q| / M
(fracción de los M picos satélite del bit presentes en el espectro).
Problema: no normaliza por la señal total del espectro — el cociente
genómico R = TPM(sat)/TPM(hub) SÍ lleva el hub en el denominador.
Diagnóstico: corr(nº de picos del espectro, nº de bits presentes)
= **+0.52** (la riqueza del espectro inflaba R_b de todos los bits);
mediana de bits presentes por molécula 3.0 (verdadero: ~46) y 5,177 de
13,748 moléculas sin ningún bit presente (37.7%).

**Versión adoptada (fiel al cociente):** R_b = |S_b ∩ Q| / |Q ∩ bins_sel|,
la fracción de los picos del espectro que son satélites del bit b.
El denominador |Q| es la "TPM del hub" (la señal total del espectro).
Con esta normalización la correlación riqueza–presentes cae a −0.22
(v1) y a **−0.04** (v2, con el piso binomial, §3.5): el confusor queda
esencialmente eliminado. Mediana de bits presentes por molécula: 27
(v1) / 22 (v2); moléculas sin ningún bit presente: 1,666 (12.1%) /
1,963 (14.3%).

### 3.4 Satélites: top-M bins por z de co-ocurrencia

Para cada bit b, sus satélites son los top-M bins por **z de
co-ocurrencia** (C − E)/√(E+0.5) frente a independencia (C = moléculas
con el bit cuyo espectro contiene el bin; E = esperado bajo
independencia), con pisos de soporte (C ≥ 10, n_b ≥ 50).

**Diagnóstico de la métrica de ranking** (primeros 256 bits, d de Cohen
de R_b entre moléculas con/sin el bit):

| Métrica | M | d mediana | d ≥ 0.5 | μ_pos | μ_neg |
|---|---|---|---|---|---|
| lift P(p\|b)/P(p\|¬b) | 20 | 0.39 | 69/256 | 0.018 | 0.002 |
| diferencia P(p\|b)−P(p\|¬b) | 20 | 0.52 | 143/256 | 0.172 | 0.087 |
| **z de co-ocurrencia** | 10 | 0.56 | 160/256 | 0.108 | 0.032 |
| **z de co-ocurrencia** | 20 | 0.61 | 187/256 | 0.098 | 0.032 |
| **z de co-ocurrencia** | 50 | 0.63 | 211/256 | 0.083 | 0.032 |
| **z de co-ocurrencia** | 100 | 0.65 | 222/256 | 0.069 | 0.030 |

El lift favorece bins raros pero poco frecuentes (μ_pos 0.018: los
satélites casi nunca aparecen); la z equilibra frecuencia y
especificidad. Se eligió **z, M=100** (la ganancia de 50→100 es pequeña
pero el coste es el mismo).

**Bug encontrado y corregido:** el producto disperso bool @ bool en
scipy es OR-lógico (0/1), NO conteo — la primera corrida dio 0 satélites
para todos los bits. Se pasó a float32 antes del producto
(`Yt.astype(float32) @ Xf.astype(float32)`).

**Baseline completo** (`10_topo_baseline.py`, 101.6 s, RSS máx 1,033 MB):
2048 bits × 100 satélites (todos con 100, 0 vacíos); d de Cohen media
0.682, mediana **0.664**, p25 0.577, p75 0.758; **1,859/2,048 bits con
d ≥ 0.5**, 82 con d ≥ 1.0, 0 con d ≥ 2.0; μ_pos media 0.082, μ_neg media
0.035; lift máximo mediano 8.33. **Nivel 1 (triage): 1,845 bits**
(n_b ≥ 200, d ≥ 0.5, lift máx ≥ 2.0). Salida:
`intermedios/topo_baseline_z_picos.npz` (1.7 MB). La versión sin
normalizar queda preservada como `topo_baseline_z_nada.npz`.

### 3.5 Predicción: dos niveles, puntaje y agregación (11_topo_predecir.py, topo_lib.py)

Por espectro de consulta: R_b para los 2048 bits (un producto disperso
contra el indicador de satélites, por bloques de 8,192 espectros);
Z_b^- = (R_b − μ_b^-)/σ_b^- y Z_b^+ = (R_b − μ_b^+)/σ_b^+;
**puntaje de presencia score_b = Φ(Z_b^- − 2.5)** (Φ = CDF normal;
score > 0.5 ⟺ Z_b^- > 2.5, el mismo umbral del agente). Agregación por
molécula: **media** de los scores de sus espectros (igual que los otros
modelos); como alternativa se calcula también el **máximo** (presencia
si ALGÚN espectro muestra el bit).

**v1 (gaussiana) vs v2 (binomial):** la v1 usa solo σ empírica. Diagnóstico:
la R_b es una fracción de conteo sesgada (masa en 0) y algunos bits
tienen σ^- diminuta (mín 0.0025), lo que infla Z de bits falsos:
tasa de falsos positivos al umbral 0.5 = **2.34%** (47 bits falsos por
molécula frente a ~7 verdaderos; precisión 13%). La **v2** usa
σ_efectiva = máx(σ_empírica, σ_binomial) con
σ_binomial = √(p0(1−p0)/n), p0 = μ^- (o μ^+), n = nº de picos del
espectro: es el test binomial de proporciones con piso por
sobredispersión. La tasa de falsos positivos baja a **1.97%** y la
correlación riqueza–presentes a −0.04. El ranking (aciertos en top-100)
apenas cambia: 10.9 (v1) → 11.1 (v2) de 46 bits reales — la calibración
mejora, pero la evidencia satélite en sí es el factor limitante (ver §6).

**Niveles y anomalías (val, mol-split, v2):** Nivel 1 = 58.4 bits
presentes por espectro (media), Nivel 2 = 64.6; moléculas sin ningún
bit presente (HUB_COLLAPSE): 1,963/13,748 (14.3%). **SATELLITE_CAPTURE
no se dispara nunca (0.0/espectro en v1 y v2):** R_b está acotada por
debajo en 0 y μ_pos es pequeña frente a σ_pos (la distribución de
presencia es muy dispersa, de 0 a ~0.3), así que un R_b bajo nunca llega
a −2.5σ por debajo de μ_pos. Es un hallazgo estructural, no un bug: el
tipo captura no se transfiere a este observable; el canal LEAPFROG hace
todo el trabajo. Se reporta así y no se fuerza.

**Tiempos:** predicción val+spec 18.7 s (v2), RSS máx 1,068 MB.

### 3.6 Señal medida (diagnóstico previo a la evaluación formal)

Val, mol-split (media de scores): bits reales 0.179 (v2) / 0.185 (v1)
frente a bits falsos 0.046 / 0.050 — **ratio ~3.7×**, la señal es real.
Aciertos en top-100: media 11.1 (v2) / 10.9 (v1), mediana 10, de 46
bits reales; en top-25: 4.9 / 4.8. Bits reales con score > 0.9: 4.9%
(v2) / 5.2% (v1); falsos: 0.38% / 0.49%. La evaluación formal (§6)
confirma el veredicto.

## 4. Control — lineal multi-salida con SGD propio (05, 06)

### 4.1 Elección y justificación

El input es el espectro bineado (200,000 bins a 0.01 Da, top-100 picos),
disperso y de muy alta dimensión; el output son 2048 bits. Se eligió un
**modelo lineal multi-salida entrenado con SGD por mini-lotes** porque:
- Es el único de los candidatos tratable con 4 CPU y ~6 GB: una pasada
  por los datos actualiza todas las salidas a la vez; la matriz de
  coeficientes W (K×2048, float32) es el único objeto grande.
- **sklearn no sirve aquí:** `SGDRegressor` solo acepta y unidimensional
  (comprobado: `ValueError: y should be a 1d array` con y (n, 2048));
  2,048 ajustes separados serían 2,048 pasadas por los datos. Por eso el
  SGD es implementación propia (mini-lotes, error cuadrático + L2,
  momento), en float32.
- ExtraTrees/MLP sobre 200k features dispersas no caben (un MLP exigiría
  densificar; ExtraTrees con 2048 salidas es inviable en tiempo y RAM).
  Un MLP pequeño sobre features reducidas (SVD) queda como trabajo futuro
  (ver §8).

### 4.2 Datos de entrenamiento (05_preparar_entrenamiento.py)

- **Un espectro por molécula de librería** (val_lib: 2,285,808 espectros
  de 261,536 moléculas): el de mayor nº de picos (nnz), determinista.
  261,536 filas; nnz medio 85.7 por espectro.
- **Selección de features: top-K bins por frecuencia de documento (df)**
  sobre esas filas. Solo 122,258 bins tienen df>0. Cobertura de nnz:
  K=10k → 87.4%, 20k → 91.9%, **50k → 98.5%**, 100k → 99.9%.
  Se eligió **K=50,000** (df mínimo incluido 17): la matriz W
  (50,000×2048 float32 = 410 MB) cabe en RAM y se conserva el 98.5% de
  la masa de picos; subir a 100k solo añadiría 1.4% de masa por el doble
  de memoria.
- **Transformación:** sqrt(intensidad) + normalización L2 por fila (la
  misma que usa el coseno del baseline 1).
- Objetivos: las huellas del pool para esas moléculas (0 sin huella).
- Tiempo: 36 s, RSS máx 1,168 MB.

### 4.3 Entrenamiento (06_entrenar_lineal.py)

- Mini-lotes de 1024, barajado por época (semilla 42), pérdida error
  cuadrático + L2 (λ=1e-4), momento 0.9, float32.
- **Pilotos de lr** (subconjunto 50k filas, 2 épocas, MSE de
  entrenamiento por época; el MSE de partida con W=0 es 0.0225):

  | lr | época 1 | época 2 | \|W\|_F |
  |---|---|---|---|
  | 0.5 | 0.01984 | 0.01807 | 36.0 |
  | 2.0 | 0.01869 | 0.01692 | 66.4 |
  | 5.0 | 0.01805 | 0.01618 | 99.1 |
  | 10.0 | 0.01762 | 0.01578 | 125.1 |

  El MSE baja con lr sin divergir hasta 10.0, con rendimientos
  decrecientes. Se eligió **lr=5.0** (equilibrio: claramente mejor que
  0.5/2.0, menos agresivo que 10.0; el MSE de entrenamiento no es el
  criterio de generalización — el veredicto lo da la validación).

- **Corrida completa:** 6 épocas sobre las 261,536 filas, lr=5.0.
  Tiempo total 2,249.2 s (~37.5 min) + predicción ~14 s = 2,263.5 s;
  RSS máx 2,230 MB.

  | Época | MSE entrenamiento | Tiempo | \|W\|_F |
  |---|---|---|---|
  | 1 | 0.01655 | 343 s | 114.7 |
  | 2 | 0.01588 | 384 s | 120.8 |
  | 3 | 0.01585 | 507 s | 121.9 |
  | 4 | 0.01584 | 343 s | 122.0 |
  | 5 | 0.01584 | 323 s | 122.0 |
  | 6 | 0.01584 | 346 s | 122.2 |

  El MSE se estanca en 0.01584 desde la época 2: el modelo convergió;
  las épocas 3–6 no aportan nada (el MSE de entrenamiento no es el
  criterio — ver §6).

- Predicción: por espectro y media por molécula (para un modelo lineal
  es equivalente a predecir el espectro medio). `P_val_lineal.npy`
  (13,748×2048, pred media 0.0171, máx 1.863),
  `P_spec_lineal.npy` (9,936×2048, pred media 0.0167, máx 1.812).

## 5. Comparación — kNN espectral → huella (07_knn_huella.py)

- Por espectro de consulta: coseno contra la librería de validación
  (ventana ±10 ppm, piso 0.002 Da, igual que el baseline 1), top-10
  moléculas por coseno, huella predicha = media ponderada por coseno de
  sus huellas; luego media por molécula sobre sus espectros.
- Idea: los análogos más cercanos comparten subestructuras. En mol-split
  la molécula NO está en la librería, así que mide exactamente eso.
- Cobertura: val — 38,377 espectros, 528 sin candidatos (1.4%), top-1
  coseno media 0.547, mediana 0.546, ≥0.9 en 17.0%; spec — 29,787
  espectros, 50 sin candidatos, top-1 coseno media 0.829, mediana 0.876,
  ≥0.9 en 43.1%. Tiempo: 182 s, RSS máx 2,438 MB (corrió en paralelo con
  el entrenamiento lineal sin OOM).
- Salidas: `P_val_knn.npy` (13,748×2048), `P_spec_knn.npy` (9,936×2048).

## 6. Validación (08_evaluar_paso2.py)

**Mol-split, semilla 42, las mismas 13,790 moléculas del baseline 1**
(13,748 con huella real; 42 sin ella se excluyen del Tanimoto y su
rango es 0 por construcción en el retrieval). Pool de retrieval: las
275,810 huellas de `huellas_pool.npz`. Métrica de decisión: **MRR@25**.

| Modelo | Tanimoto cont. media | Tanimoto cont. mediana | Tanimoto bin100 media | MRR@25 | Top-1 | Top-5 | Recall@25 |
|---|---|---|---|---|---|---|---|
| Nulo (frecuencia media) | 0.2207 | 0.2263 | 0.1728 | — | — | — | — |
| kNN | **0.3148** | 0.2849 | **0.2176** | **0.0346** | 0.0014 | 0.0752 | 0.1555 |
| Topológico v1 (gaussiano, media) | 0.1001 | 0.0934 | 0.0830 | 0.0140 | 0.0067 | 0.0199 | 0.0551 |
| Topológico v1 (gaussiano, máx) | 0.0963 | 0.0899 | 0.0809 | 0.0138 | 0.0063 | 0.0202 | 0.0535 |
| Topológico v2 (binomial, media) | 0.1020 | 0.0959 | 0.0850 | 0.0148 | 0.0072 | 0.0212 | 0.0561 |
| Topológico v2 (binomial, máx) | 0.1007 | 0.0946 | 0.0842 | 0.0148 | 0.0068 | 0.0222 | 0.0563 |
| Lineal (control) | 0.2475 | 0.2412 | 0.2421 | 0.0085 | 0.0032 | 0.0116 | 0.0410 |

**Tabla completa (mol-split, 13,748 moléculas).** La agregación por máximo
no ayuda: v2 máx = v2 media (0.0148), v1 máx < v1 media (0.0138 <
0.0140). La v2 binomial ≈ v1 gaussiana en MRR (0.0148 vs 0.0140): la
calibración mejora, el ranking no. El topológico queda 2.3× por debajo
del kNN en MRR@25 y 2.8× en Recall@25, aunque gana 5× en Top-1
(0.0072 vs 0.0014): comportamiento picudo — acierta fuerte en pocas
moléculas, falla en la mayoría. El lineal es el peor en MRR (0.0085)
pese al mejor bin100.

**Lectura clave: el bin100 NO predice el MRR.** El lineal tiene el mejor
Tanimoto bin100 (0.2421, por encima del kNN 0.2176) pero el PEOR MRR@25
(0.0085): sus scores son densos y salen de [0,1] (máx 1.863), así que el
Tanimoto continuo contra la huella real se diluye con la masa puesta en
bits falsos. El kNN concentra su masa (media ponderada de 10 vecinos:
a lo sumo ~460 bits con score no nulo), así que su Tanimoto continuo es
más limpio y su retrieval mejor. **El ranking por MRR@25 (la métrica del
concurso) es: kNN 0.0346 > topo v2 0.0148 > topo v1 0.0140 > lineal
0.0085.** El ganador para el envío es el **kNN**. El topológico, pese a
tener el mejor Top-1 (0.0072 vs 0.0014 del kNN), queda 2.3× por debajo
en MRR@25.

Oráculo (huella real como consulta, val): MRR@25 0.9684, Top-1 0.9612,
Recall@25 0.9916 — el techo del retrieval con el pool actual (limitado
por colisiones exactas de huella: media 1.64 gemelas por molécula,
mediana 0, 95.4% sin gemelas, máx 359). Baseline 1 en mol-split: 0.0000
por construcción.

**Spec-split no medido para el paso 2.** El retrieval en spec-split es
solo referencia (optimista: las mismas moléculas están en train) y no
cambia la decisión — el mol-split es el criterio. Las predicciones spec
(`P_spec_*.npy`) existen; medir el retrieval spec-split queda como
trabajo futuro si hace falta el número de referencia. El baseline 1 en
spec-split era MRR@25 0.8691 (búsqueda en librería, no huella).

**Incidente 2026-11-09 ~16:25 (muerte por memoria):** las evaluaciones
`topo_val_max` y `lineal_val` corrían EN PARALELO (dos cadenas) y ambas
murieron abruptamente sin traceback (SIGKILL) a las 16:24:37 y 16:24:55,
con el host en 191 MB libres — el OOM killer del host. `topo_val_max`
iba al 92% del retrieval (13,056/13,748); `lineal_val` al 2% (256/13,748).
Los números de Tanimoto del lineal (0.2475/0.2421) son válidos: se
calculan ANTES del retrieval. Lección (la misma del baseline 1): **una
evaluación pesada a la vez, nunca dos en paralelo.** Las evaluaciones
pendientes se relanzaron en cadenas secuenciales y completaron.

**Incidente 2026-11-09 ~16:37 (archivos de códigos de molécula):** la
cadena secuencial falló dos veces por nombres de archivo: los códigos
de molécula de las agregaciones por máximo nunca se guardaron como
archivos propios (`P_val_mol_topo_max.npy`,
`P_val_mol_topo_v2_max.npy` no existían) y el archivo de predicciones
v2-máx se llama `P_val_topo_max_v2.npy`, no `P_val_topo_v2_max.npy`
(la convención de nombres es inconsistente: `max` antes de `v2`).
Se resolvió copiando los códigos de la media (la media y el máximo se
agregan de los mismos espectros en el mismo orden, así que los códigos
son idénticos) y usando el nombre correcto. Las dos evaluaciones
completaron después.

## 7. Test y envío (09_predecir_test_paso2.py)

**Modelo elegido: kNN** (mejor MRR@25 en mol-split, 0.0346; ver §6).
`09_predecir_test_paso2.py --modelo knn`, 52.9 s, RSS máx 2,911 MB.

- Test: 1,213 espectros bineados (99,187 nnz, 0 descartados), 400
  moléculas, 3.03 espectros/molécula (media). Predicción kNN por
  espectro: 0 espectros sin candidatos. Huella por molécula = media de
  las huellas predichas de sus espectros.
- Retrieval: 400 consultas × pool 275,810 (Tanimoto continuo).
- **`salida/submission_paso2.csv` (knn): 400 filas, orden de
  `molecule_id` idéntico a `datos/sample_submission.csv`, 25 SMILES por
  fila en las 400 (media 25.0, mín 25, máx 25), 0 vacías, 0 fallback
  "CCO", 0 NaN, 0 SMILES duplicados dentro de una fila.** md5
  `ed758b99cdb3a91ac690f02fc3618779`. Verificado de forma independiente
  (no solo por el log del script).
- **`salida/submission.csv` (baseline 1) intacto:** md5
  `7a7da389417fb162039a94394d76e1ee` verificado antes y después de la
  corrida (el script lo comprueba al inicio y al final).

**Cobertura de la librería en el test (hallazgo principal):** el top-1
coseno contra `full_lib` tiene **media 0.9999999956 y mediana 1.0**;
**400/400 moléculas con top-1 ≥ 0.95** (y ≥ 0.9, ≥ 0.8, ≥ 0.7). Es decir,
**las 400 moléculas del test ya están en la librería con un espectro
esencialmente idéntico**. Para ESTE test, la búsqueda en librería
(baseline 1) es la vía fuerte y el modelo de huellas es un complemento:
el solape entre el top-25 del paso 2 (kNN) y el top-25 del baseline 1
es solo **media 2.6, mediana 2, cero en 0/400** — los dos métodos
ordenan distinto; el baseline 1 pone arriba al candidato de coseno
1.0 (casi seguro la molécula verdadera), mientras el kNN ordena por
Tanimoto a una huella predicha aproximada. **El valor del modelo de
huellas no está en este test, sino en moléculas realmente ausentes de
la librería (clases 2/3 en general).** El envío paso 2 se entrega como
segundo modelo, no como sustituto del baseline 1.

**Nada enviado a Kaggle.** La decisión de enviar (y qué envío) es de
Elán.

## 8. Limitaciones honestas y siguiente paso

### 8.1 Qué no mide este paso

- **El pool NO es PubChem/COCONUT.** El pool de retrieval es
  `mapa_smiles_full.parquet`: los 275,810 códigos de molécula de la
  librería del concurso (train + val + test + extras), de los que
  275,303 tienen huella. Es la simulación de clases 2/3 disponible
  en local (la molécula es nueva para el modelo, su huella está en el
  pool), pero es **optimista** frente al concurso real: PubChem/COCONUT
  tienen millones de moléculas, y un pool mayor hace el retrieval más
  difícil (más candidatos que pueden quedar por encima de la verdadera).
  El oráculo (MRR@25 0.9684) es el techo con ESTE pool, no con el pool
  real.
- **Tautómeros y estereoquímica.** La identidad de molécula es el
  inchikey14 (primer bloque del InChIKey, tras canonicalización
  tautomérica con RDKit). La métrica no distingue tautómeros ni
  estereoisómeros que compartan inchikey14, y el modelo tampoco: la
  huella de Morgan sí distingue algunos, así que la huella predicha
  puede apuntar a un estereoisómero distinto del verdadero y aun así
  puntuar por inchikey14. El modelo no resuelve estructura fina.
- **Aductos.** El modelo ignora el aducto y la masa del precursor: todos
  los espectros de una molécula se agregan por media sin distinguir
  modo de ionización. La masa del precursor (fórmula molecular) es
  evidencia fuerte que no se usa. No hay emparejamiento entre aductos.
- **Tope de 100 picos.** Solo los top-100 picos por espectro (bins
  0.01 Da) entran al modelo; los fragmentos de baja intensidad se
  descartan aunque puedan llevar evidencia de subestructura.
- **507 moléculas sin espectro válido** no pueden ser candidatas (están
  en el mapa pero sin espectro; el pool tiene 275,303 huellas de
  275,810 códigos).
- **Spec-split es optimista.** En spec-split las mismas moléculas están
  en train y val (otros espectros): mide memoria espectral, no
  generalización a moléculas nuevas. El número honesto para clases 2/3
  es el mol-split.
- **Un solo split.** Todos los números son de un único mol-split
  (semilla 42, las 13,790 moléculas del baseline 1). No se midió la
  varianza entre splits.
- **El canal SATELLITE_CAPTURE no existe en este observable** (§3.5):
  el modelo topológico solo tiene un canal de trabajo (LEAPFROG). No se
  forzó un segundo canal.
- **El modelo topológico es peor que el nulo en Tanimoto** (0.1020 frente
  a 0.2207): su huella predicha es menos parecida a la real que predecir
  la frecuencia media de cada bit. El kNN (0.3148) y el lineal (0.2475)
  sí superan al nulo. Se reporta así; la adaptación topológica, tal como
  está construida, no aprende una huella usable — la evidencia satélite
  (top-100 bins por bit) es el factor limitante, no solo la calibración
  de Z (la v2 mejora la calibración sin apenas cambiar el ranking).
- **El lineal no es una probabilidad.** Sus predicciones salen de
  [0,1] (máx 1.863): es una regresión, no una probabilidad calibrada.
  Sirve para ordenar (Tanimoto continuo y retrieval), no como
  confianza.
- **El envío real depende de Kaggle.** La simulación local usa
  inchikey14; el emparejamiento real en Kaggle (canonicalización,
  tautómeros) puede diferir ligeramente, y el MRR@25 real se mide
  contra el pool completo de la competencia, no contra este pool.

### 8.2 Siguiente paso

**Veredicto:** el ganador es el **kNN (MRR@25 0.0346)** y el envío paso 2
ya existe (`salida/submission_paso2.csv`, knn). El modelo topológico
(MRR@25 0.0148) y el lineal (0.0085) quedan por debajo; la adaptación
topológica, tal como está, no aprende una huella usable y no se fuerza.

**Próximos pasos, en orden de coste:**
1. **Combinar librería + huella en un solo envío.** El test está
   400/400 cubierto por la librería (§7): el envío fuerte es el
   baseline 1 arriba y el paso 2 (kNN) como respaldo para moléculas
   sin match espectral. Medir en mol-split cuánto gana la combinación
   (p. ej. top-25 = top-k de librería + resto por huella).
2. **Usar la masa del precursor y el aducto** como evidencia adicional
   (fórmula molecular) antes del retrieval — es la evidencia más
   fuerte que hoy se ignora.
3. **Mejorar el kNN:** más vecinos, ponderación por coseno al cuadrado,
   o un modelo aprendido con salidas dispersas (el lineal denso pierde
   en retrieval aunque gane en bin100). Un MLP pequeño sobre features
   reducidas (SVD de los 50k bins) como tercer modelo.
4. **Ampliar el pool** de retrieval con COCONUT/PubChem si hay acceso
   (el pool actual es solo la librería del concurso).
5. **Para el topológico,** revisar el observable R_b (p. ej. usar
   intensidades en vez de presencia de pico, o un score por bit que no
   dependa de un umbral fijo de Z) antes de insistir en la calibración
   de Z — la evidencia satélite es el factor limitante, no la
   calibración.

## 9. Archivos

- `04_huellas.py` — huellas del pool.
- `05_preparar_entrenamiento.py` — 1 espectro/molécula + top-K bins.
- `06_entrenar_lineal.py` — SGD lineal multi-salida + predicción val/spec.
- `07_knn_huella.py` — kNN espectral → huella.
- `08_evaluar_paso2.py` — Tanimoto + retrieval + oráculo + colisiones.
- `09_predecir_test_paso2.py` — predicción del test y submission_paso2.csv
  (modelos topo/lineal/knn).
- `10_topo_baseline.py` — red de referencia topológica (satélites por z
  de co-ocurrencia + μ/σ por bit; métricas lift/diferencia/z; M; normalización
  picos/nada).
- `11_topo_predecir.py` — predicción topológica val/spec (media y máx).
- `topo_lib.py` — cargar baseline, indicador de satélites, predicción
  (scoring gaussiano/binomial), agregación por máximo.
- `paso2_lib.py` — pool, Tanimoto continuo, retrieval.
- `intermedios/` — `huellas_pool.npz`, `huellas_freq.npz`,
  `X_train_paso2.npz`, `y_train_paso2.npz`, `train_mol_paso2.npy`,
  `bins_sel_paso2.npy`, `W_lineal_paso2.npy`, `P_val_lineal.npy`,
  `P_spec_lineal.npy`, `P_val_knn.npy`, `P_spec_knn.npy`,
  `topo_baseline_z_picos.npz` (+ `topo_baseline_z_nada.npz`, versión
  sin normalizar preservada; `topo_baseline_*_muestra.npz` de diagnóstico),
  `P_val_topo.npy`, `P_val_topo_max.npy`, `P_spec_topo.npy`,
  `P_spec_topo_max.npy`, `P_val_topo_v2.npy`, `P_val_topo_max_v2.npy`,
  `P_spec_topo_v2.npy`, `P_spec_topo_max_v2.npy`,
  `P_val_mol_topo.npy`, `P_val_mol_topo_v2.npy`,
  `P_val_mol_topo_max.npy`, `P_val_mol_topo_v2_max.npy` (copias de los
  códigos de la media; la media y el máximo se agregan de los mismos
  espectros en el mismo orden),
  `diag_val_topo.npz`, `diag_spec_topo.npz`,
  `diag_val_topo_v2.npz`, `diag_spec_topo_v2.npz`,
  `P_test_knn.npy`, `top_idx_test_knn.npy`, `test_q_paso2.npz`.
- `verificacion/` — `huellas_stats.json`, `preparar_paso2.json`,
  `entrenar_lineal.log` (6 épocas), `knn_paso2.json`,
  `topo_baseline*.json` (diagnóstico de métricas y M),
  `topo_baseline_picos.log`, `topo_predecir.json`,
  `topo_predecir_v2.json`, `evaluar_knn_val.log`,
  `evaluar_topo_val.log`, `evaluar_topo_val_v2.log`,
  `evaluar_lineal_val.log`, `evaluar_topo_val_max.log`,
  `evaluar_topo_val_max_v2.log`, `predecir_test_knn.log`,
  `evaluar_val_secuencial.done`, `evaluar_max_val.done`,
  `metricas_paso2_knn_val.json`, `metricas_paso2_topo_val.json`,
  `metricas_paso2_topo_val_v2.json`, `metricas_paso2_lineal_val.json`,
  `metricas_paso2_topo_val_max.json`,
  `metricas_paso2_topo_val_max_v2.json`,
  `ranks_paso2_*.npy`, `top_idx_paso2_*.npy`,
  `test_paso2_knn.json`.
- `salida/submission_paso2.csv` — 400 filas, modelo kNN
  (md5 `ed758b99cdb3a91ac690f02fc3618779`).
- `salida/submission.csv` — baseline 1, **intacto**
  (md5 `7a7da389417fb162039a94394d76e1ee`, verificado antes y después).
