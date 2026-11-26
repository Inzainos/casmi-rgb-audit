# CASMI — Informe Morgan: filtro de masa, binarización WPC y submission

**Fecha:** 2026-11-10 (fase completa: ~11:33–12:30 CST, con GO de Elán "Arranca")
**Repo:** `~/workspace/enveda-casmi-2026/` (commit abf4976 + trabajo local, sin push)
**Reglas:** nada a Kaggle, nada a GitHub/main sin decisión de Elán; todo local;
logs de corrida en `verificacion/` (AGENTS.md regla 5); los tres submissions
existentes no se tocan (md5 verificados antes y después de cada paso).

## 0. Resumen ejecutivo (números medidos, no estimados)

El modelo espectro → huella Morgan **sí sube el score público esperado, y por
mucho**: el filtro de masa de precursor multiplica el MRR@25 mol-split del
lineal por 43.7× (0.0085 → 0.3719) y el de la variante lineal_bin lo deja en
**0.4146**, el mejor modelo Morgan medido. El submission nuevo
`salida/submission_morgan.csv` usa ese modelo. Su MRR esperado en el test
oculto es **≈ 0.416** (0.17 × 0.4250 spec + 0.83 × 0.4146 mol), frente al
**0.146 público actual** (baseline 1 solo). El blend baseline-1 + Morgan
(regla top-1 coseno ≥ 0.95) estima **≈ 0.462** si el oculto se comporta como
el test visible, pero **≈ 0.400** si se comporta como mi validación
spec-split — el blend es decisión de Elán, no de este informe (§7).

La Idea 1 WPC (binarizar el espectro) **ayuda de forma real pero pequeña**:
+12.6% a presupuesto piloto, +0.95% a presupuesto completo; no supera al
lineal_bin K=50k existente. Éxito por el criterio del documento WPC
(MRR mol-split > kNN 0.0346), pero sin coronar.

| Modelo (mol-split, filtro de masa) | MRR@25 | Top-1 | Recall@25 |
|---|---|---|---|
| nulo (frecuencia) | 0.0981 | 0.0423 | 0.3899 |
| oráculo (huella real) | 0.9937 | 0.9901 | 1.0000 |
| kNN | 0.1406 | 0.0218 | 0.6111 |
| lineal | 0.3719 | 0.2497 | 0.7700 |
| **lineal_bin (clip [0,1] + top-100)** | **0.4146** | 0.2908 | 0.8014 |
| topo_v2 | 0.4123 | 0.2780 | 0.8427 |
| ens (media knn+lineal+topo) | 0.2761 | 0.1026 | 0.8066 |
| rrf | 0.2968 | — | 0.8449 |
| bin005_full_bin (WPC, K=20k) | 0.4023 | 0.2830 | 0.7827 |

Referencias sin filtro (informe-paso2): kNN 0.0346, topo_v2 0.0148, lineal
0.0085, oráculo 0.9684. Baseline 1: spec-split 0.8691, mol-split 0.000 por
construcción. Público (envío 3, baseline 1): **0.146**.

## 1. Entorno (corrección aplicada en esta fase)

El brief inicial decía venv py313; el AGENTS.md del repo manda el **Python
3.12.3 del sistema** (numpy 1.26.4, pandas 2.1.4, scipy 1.11.4 — nunca
subir numpy a 2.x). La caja había reseteado los paquetes pip, así que se
reinstalaron con `pip --break-system-packages` (log
`verificacion/pip_instalacion_sistema.log`):

- scikit-learn 1.6.1, pyarrow 17.0.0, **rdkit 2024.03.6** (pin del repo).
- numpy quedó en 1.26.4 (pip lo dejó intacto).

**Validación RDKit** (`validar_huellas_rdkit.py`, muestra de 500 SMILES del
pool, stride 551, mismo código que `04_huellas.py`:
`GetMorganGenerator(radius=2, fpSize=2048)` + `MolFromSmiles`):

- RDKit 2024.03.6 (sistema) vs 2026.9.1 (py313-venv): **0 bits distintos
  de 1,024,000 (0.00000%)**, 0 moléculas distintas, 0 fallos de parseo en
  ambas. Bits/mol media 45.5 en ambas.
- Pool `huellas_pool.npz` vs recomputo 2024.03.6: **0 bits distintos**.
- Veredicto: **IDÉNTICAS** (`verificacion/validacion_rdkit.json`,
  `verificacion/validacion_rdkit.log`). El pool es consistente con el pin
  del repo; todo el pipeline de aquí en adelante usa el sistema +
  RDKit 2024.03.6.

**Nota de trazabilidad:** `12_masas.py` y `13_evaluar_masa.py --split mol`
corrieron bajo py313-venv (numpy 2.5.4) antes de la corrección de entorno;
ninguno usa RDKit, así que sus resultados no se ven afectados. Todo lo
demás (13 spec, 14, 15, 16, 17) corrió bajo el Python del sistema.
Cada corrida registra su entorno en su log/JSON.

**RAM de la caja durante la fase:** ~190–545 MB libres, ~640–1,370 MB
disponibles (trabajo cuántico en paralelo). Un solo proceso pesado a la
vez; chunks, float32, memmap. Los tres entrenamientos completos cupieron
(RSS máx 866–1,037 MB) gracias a la actualización SGD in-place de 14
(sin temporales de 164/410 MB por operación).

**Rutas:** los scripts 04–08 llevan `BASE =
/home/hatch/workspace/snt-taxonomia/kaggle-casmi`, pero el workspace vivo
es `~/workspace/enveda-casmi-2026/`. Los scripts nuevos (12–17) usan el
BASE correcto; los viejos **no se ejecutaron** contra el path viejo. Los
tres logs que quedaron en `logs/` en la primera fase (`12_masas`,
`13_evaluar_masa_mol`, `rdkit_install`) se movieron a `verificacion/` al
cerrar esta fase (AGENTS.md regla 5) y el directorio `logs/` se eliminó.

## 2. El filtro de masa de precursor (scripts 12 y 13)

**Idea:** sin hacer ninguna matemática de aductos (los aductos del train
son un zoo: [M+H]+ 67,344; [2M+Na]+ 21,021; [M-H]- 20,712; [2M+H]+ 9,809;
[2M-H]- 5,952; dímeros ≈ 28% de la muestra; `val_qmol_meta.npz` ni
siquiera guarda el aducto), el filtro usa solo masas: candidatos = moléculas
con algún espectro en `full_lib` (2,438,064 espectros) dentro de ±10 ppm
(piso 0.002 Da, igual que el baseline 1) de cualquiera de los precursores
de la consulta. Todas las moléculas mol-split val están en `full_lib`.

**`12_masas.py`** (5.5 s, RSS máx 107 MB, `verificacion/12_masas.log`):
construye `intermedios/precursores_full_idx.npz` (precursores ordenados +
códigos de molécula) y `intermedios/query_prec_val_mol.npz`,
`query_prec_val_spec.npz`, `intermedios/query_prec_test.npz`
(`verificacion/masas_stats.json`).

| Split (ppm 10) | candidatos/mol mediana | media | p90 | máx | verdad sobrevive |
|---|---|---|---|---|---|
| val_mol (13,748) | 106 | 141 | 320 | 1,001 | 1.0000 (13,748/13,748) |
| val_spec (9,936) | 126 | — | — | — | 1.0000 |
| test (400) | 126 | 170 | — | 841 | — (verdad desconocida) |
| val_mol ppm 20 | 171 | — | — | — | — |

**`13_evaluar_masa.py`** (mol: 44 s, RSS máx 443 MB; spec: 40 s, RSS máx
395 MB): retrieval con filtro de masa, una pasada, P por memmap, 8
variantes. Resultados en `verificacion/metricas_masa_val_mol.json` y
`verificacion/metricas_masa_val_spec.json` (+ `ranks_masa_*` y
`top_idx_masa_*` por variante).

### 2.1 Mol-split (13,748 moléculas, pool 275,810, mediana 106 candidatos)

| Variante | MRR@25 | Top-1 | Top-5 | Recall@25 | Tanimoto |
|---|---|---|---|---|---|
| nulo | 0.0981 | 0.0423 | — | 0.3899 | — |
| oráculo | 0.9937 | 0.9901 | — | 1.0000 | 1.0000 |
| knn | 0.1406 | 0.0218 | — | 0.6111 | — |
| lineal | 0.3719 | 0.2497 | — | 0.7700 | — |
| **lineal_bin** | **0.4146** | **0.2908** | **0.5647** | 0.8014 | — |
| topo_v2 | 0.4123 | 0.2780 | 0.5732 | 0.8427 | 0.0972 |
| ens | 0.2761 | 0.1026 | — | 0.8066 | — |
| rrf | 0.2968 | — | — | 0.8449 | — |

Lectura: el filtro solo ya levanta hasta el nulo (0.0981 > kNN sin filtro
0.0346). El lineal pasa de peor (0.0085) a segundo mejor (0.3719, 43.7×);
lineal_bin es el mejor (0.4146). topo_v2 queda segundo (0.4123) con el
mejor Recall@25 entre modelos simples salvo rrf. El ensamble media y el
RRF **empeoran** frente a lineal_bin (mezclar knn, que es débil, arrastra
el ranking). El oráculo 0.9937 (antes 0.9684) es el techo con este filtro
y este pool.

### 2.2 Spec-split (9,936 moléculas, mediana 126 candidatos)

| Variante | MRR@25 | Top-1 | Top-5 | Recall@25 |
|---|---|---|---|---|
| nulo | 0.0747 | 0.0289 | 0.1080 | 0.3416 |
| oráculo | 0.9963 | 0.9940 | 0.9995 | 1.0000 |
| knn | 0.8411 | 0.7608 | 0.9442 | 0.9892 |
| lineal | 0.3735 | 0.2499 | 0.5191 | 0.7789 |
| lineal_bin | 0.4250 | 0.3013 | 0.5701 | 0.8079 |
| topo_v2 | 0.4000 | 0.2659 | 0.5549 | 0.8410 |
| ens | 0.8186 | 0.7313 | 0.9305 | 0.9849 |
| rrf | 0.5811 | 0.4404 | 0.7509 | 0.9848 |

Lectura: en moléculas en librería el kNN Morgan (0.8411) queda apenas por
debajo del baseline-1 coseno (0.8691, sin filtro de masa extra — el 03 ya
filtra por precursor). El ensamble (0.8186) se acerca al kNN. El lineal_bin
(0.4250) es el mejor lineal también aquí, y es el número que usa el
submission Morgan para la fracción clase-1 del oculto (§6).

## 3. WPC Idea 1: binarizar el espectro (scripts 14 y 15)

**Hipótesis** (`wpc-schumann-ideas.md` §3, Idea 1): el espectro como "foto"
WPC — pico presente (1) o ausente (0) por encima de un umbral τ. La huella
objetivo ya es binaria; las features continuas (sqrt de intensidad + L2)
pueden estar ordenando mal el ranking (el lineal tenía el mejor Tanimoto
bin100, 0.2421, pero el peor MRR, 0.0085). Criterio de éxito del documento:
MRR@25 mol-split > kNN 0.0346. Métrica que decide: MRR@25, no Tanimoto.

**Implementación** (`14_entrenar_lineal_binario.py`, variante de 06 con
BASE corregido): `X_bin = (X > τ)` sobre los valores sqrt+L2 guardados en
`X_train_paso2.npz`, y luego L2 por fila (misma transformación para
entrenar y predecir: primero sqrt+L2 como en 06, luego umbral, luego L2;
τ vive en la escala sqrt+L2). El control es el mismo script sin
`--binario`. Evaluación con `15_evaluar_pred.py` (mismo protocolo de
filtro de masa que 13, para un P arbitrario, crudo o con variante bin).

**Decisiones de diseño registradas:** K=20,000 columnas (top por df dentro
de las 50k de 05) porque W+V a K=50k son 820 MB y no caben con la RAM
actual; actualización SGD in-place (`np.multiply/divide` con `out=`) para
no crear temporales de 164 MB por operación. Subset piloto = primeras
80,000 filas (códigos de molécula más bajos, igual que 06); el control y
los binarizados usan el mismo subset, así que el sesgo se cancela en la
comparación.

### 3.1 Piloto (K=20k, 80k filas, 3 épocas, lr 0.5, mom 0.9, lam 1e-4)

| Modelo | τ | picos que quedan | MSE final | MRR crudo | MRR bin |
|---|---|---|---|---|---|
| ctrl20k (122 s, RSS 889 MB) | — | 100% | 0.01725 | 0.2477 | 0.2886 |
| bin005 (61 s, RSS 794 MB) | 0.05 | 19.0% | 0.01660 | 0.2809 | **0.3251** |
| bin010 (51 s, RSS 773 MB) | 0.10 | 7.4% | 0.01718 | 0.2587 | 0.3053 |

A presupuesto fijo la binarización gana: bin005 supera al control +13.4%
en crudo (0.2809 vs 0.2477) y +12.6% en bin (0.3251 vs 0.2886). τ=0.05 >
τ=0.10. La variante bin de salida (clip + top-100) ayuda a los tres,
igual que en el lineal full.

### 3.2 Presupuesto completo (K=20k, 261,536 filas, 6 épocas)

| Modelo | Tiempo | RSS máx | MSE final | \|W\|_F | MRR crudo | MRR bin |
|---|---|---|---|---|---|---|
| bin005_full | 342 s | 866 MB | 0.01569 | 94.9 | 0.3564 | **0.4023** |
| ctrl20k_full | 782 s | 1,037 MB | 0.01605 | 101.3 | 0.3546 | 0.3985 |

A presupuesto completo la binarización sigue ganando, pero solo +0.95%
relativo en bin (0.4023 vs 0.3985) y +0.5% en crudo. La ganancia del
piloto era sobre todo eficiencia muestral: con todos los datos y 6 épocas
el modelo sqrt alcanza casi lo mismo. **Ninguno supera al lineal_bin K=50k
existente (0.4146)**: K=20k cuesta más de lo que la binarización gana.

**Veredicto Idea 1:** éxito por el criterio del documento (0.4023 ≫ 0.0346),
mejora real y medida, pero **no es el mejor modelo Morgan**. El submission
usa lineal_bin K=50k. Queda como opción documentada: un lineal binarizado
a K=50k ganaría ~+1% sobre 0.4146 si la ganancia se mantiene, pero su
entrenamiento no cabe en esta caja con momento (W+V+grad+tmp ≈ 1.64 GB);
una variante sin momento y con decaimiento desacoplado cabría (~1.06 GB)
— no probada, ganancia esperada ~+0.004 de MRR, que en el oculto vale
~+0.003. No se hizo: costo/beneficio no compensa ahora.

**Ideas WPC 2–5 (no intentadas en esta fase):** conteos por banda m/z
(Idea 2), barrido τ × K como calibración (Idea 3 — parcialmente cubierto:
τ ∈ {0.05, 0.10} a K=20k), re-ranking por perfil binario Jaccard → Tanimoto
(Idea 4), resúmenes compactos precomputados (Idea 5, proceso). Ver
`wpc-schumann-ideas.md` §3.

## 4. Hallazgo de normalización (05/06) — medido, no afecta los MRR

Al validar el blend (§7) apareció un máximo de coseno 1.0495 > 1.0, señal
de que algo no era un coseno. Medido en `16_cobertura_val.py`: la "L2" de
`05_preparar_entrenamiento.py` y `06_entrenar_lineal.py` usa
`norms = sqrt(Σ sqrt(int))` en vez de la norma L2 verdadera
`sqrt(Σ int)`. Consecuencia: los vectores **no son unitarios** — norma L2
verdadera media de `X_train`: **0.595** (mediana 0.587); de las consultas
tras transformar: 0.637 (mol) / 0.644 (spec).

**No afecta a los resultados Morgan:** entrenar y predecir usan la misma
transformación, así que todos los MRR de este informe son válidos como
medidos; la normalización es solo un escalado de features, consistente en
todo el camino. Sí afecta a cualquier cosa que compare contra un umbral de
coseno verdadero (como la regla de blend ≥ 0.95): para eso el 16
renormaliza cada vector a norma L2 verdadera antes del dot. Queda como
mejora futura sin probar: corregir 05/06 a L2 verdadera y re-medir el MRR
(efecto desconocido).

## 5. Submission Morgan (`17_generar_submission_morgan.py`)

**Modelo:** lineal K=50k (`W_lineal_paso2.npy`, 410 MB) + variante
lineal_bin + retrieval con filtro de masa ±10 ppm (mismo protocolo que 13).
Sin baseline-1 ni blend (el blend es decisión de Elán).

**Archivo:** `salida/submission_morgan.csv` — 400 filas × 25 SMILES,
md5 `854584724275387743e744fd77757961` (401 líneas con encabezado).
Generado en 4 s, RSS máx 521 MB (`verificacion/17_generar_submission_morgan.log`,
`verificacion/submission_morgan.json`).

**Detalles medidos:**
- Predicción test: `intermedios/P_test_lineal.npy` (400 × 2048, media
  0.0196, máx 2.534), agregada por molécula (media de espectros, 3.03
  espectros/mol). Retrieval top-40 por molécula, luego 25 SMILES únicos.
- Candidatos/mol en test: mediana 126, media 169.9, mín 2, máx 841.
- 33/400 filas necesitaron relleno (+359 repeticiones del top-1, regla de
  `11_rellenar_25.py`); la peor es la de 2 candidatos. 0 códigos sin SMILES
  saltados.
- Solape top-25 con baseline-1: media 9.2/25, mediana 8, cero en 3/400,
  idéntico (25/25) en 2/400. El ranking Morgan difiere del espectral como
  se espera, pero comparte ~37% de candidatos.
- Formato verificado: 400 filas en el orden de `sample_submission.csv`,
  25 SMILES cada una, sin vacíos, sin fallback CCO, sin duplicados salvo
  el relleno documentado.
- Los tres submissions existentes quedaron intactos (md5 verificados
  antes y después: `7a7da389…`, `ed758b99…`, `7f082916…`).

**MRR esperado en el oculto (Morgan solo):** 0.17 × 0.4250 (spec-split
lineal_bin) + 0.83 × 0.4146 (mol-split lineal_bin) = **≈ 0.416**, frente a
0.146 público actual. Ver §6 por las hipótesis detrás de este número.

## 6. Qué estima cada número (lectura honesta)

- **El proxy local mol-split es optimista:** el pool contiene la molécula
  verdadera (simula tener PubChem); las moléculas nuevas del oculto NO
  están en la librería de train. La ganancia real depende de que el
  oculto tenga isómeros/tautómeros que compartan inchikey14 y de la
  fracción clase-1 (~17%). El conjunto de candidatos real tras filtrar
  por masa en PubChem (millones de moléculas) sería mucho mayor que ~106.
- **El test visible es 400/400 en librería** (top-1 coseno media 1.000),
  así que el submission Morgan puntuaría PEOR que baseline-1 en el test
  visible. El público (0.146) viene del oculto; el oculto es lo que manda.
- **Cualquier envío mejorado debe ser también un notebook en runtime:**
  un submission.csv estático falla en el re-run (el oculto reemplaza al
  visible; envíos 1–2). Morgan en un notebook Kaggle tiene restricciones:
  no hay RDKit en la imagen (las huellas del pool tendrían que viajar
  como dataset); el kNN necesita construir la librería en runtime (~5 min,
  cabe). Para el lineal, los artefactos a embarcar son W (410 MB) + las
  huellas del pool + el código de bineado y retrieval; la corrida local
  del submission completo tarda 4 s una vez cargado W. **No se construyó
  el notebook en runtime en esta fase** — es el siguiente paso si Elán
  decide enviar.
- **El oráculo 0.9937 no es alcanzable:** es el techo con este filtro y
  este pool; mide cuánto margen queda en la calidad de la huella predicha.

## 7. Blend baseline-1 + Morgan (medición, NO decisión)

La vía paralela pendiente de decisión de Elán es el blend con la regla
"top-1 coseno ≥ 0.95 → top-25 de baseline-1, si no → top-25 Morgan con
filtro de masa". Aquí está la medición que faltaba: la fracción de
enrutamiento, con coseno verdadero (`16_cobertura_val.py`, contra el
espectro representativo max-nnz de cada molécula candidata; el top-1 real,
máximo sobre todos los espectros, es ≥ al medido).

| Split | top-1 coseno media | mediana | p90 | ≥ 0.95 | ≥ 0.90 | ≥ 0.99 |
|---|---|---|---|---|---|---|
| mol (nuevas, n=13,748) | 0.574 | 0.587 | 0.944 | **8.82%** (1,212) | 16.58% | 0.81% |
| spec (en librería, n=9,936) | 0.837 | 0.870 | 0.973 | **19.30%** (1,918) | 39.04% | 3.72% |

**Interpretación:**
- **Mol-split ≥ 0.95 = 8.82% es la tasa de enrutamiento FALSO:** moléculas
  nuevas con un análogo ≥ 0.95 en la librería, que irían a baseline-1 y
  recibirían una molécula equivocada (puntaje 0 en esa consulta; Morgan
  la habría resuelto ~41% de las veces). Es una cota inferior (el top-1
  real es mayor que el medido contra representantes).
- **Spec-split ≥ 0.95 = 19.30% es bajo porque las consultas son
  adquisiciones distintas** de la misma molécula (el representativo en la
  librería es otro espectro). En el test visible, en cambio, el top-1 es
  1.0 exacto en 400/400: misma adquisición. Si el oculto está construido
  como el visible (misma adquisición para clase-1), la clase-1 se enruta
  ~100%; si está construido como mi spec-split val (adquisición distinta),
  solo ~19%.
- **Estimaciones del blend** (f_clase1 = 0.17, baseline-1 spec 0.8691,
  Morgan lineal_bin mol 0.4146 / spec 0.4250):
  - Enrutamiento ideal (clase perfecta): 0.17×0.8691 + 0.83×0.4146 = **0.492**.
  - Regla ≥ 0.95, oculto como el visible: 0.17×0.8691 +
    0.83×(0.088×0 + 0.912×0.4146) = **≈ 0.462**.
  - Regla ≥ 0.95, oculto como mi spec-split val: 0.17×(0.19×0.8691 +
    0.81×0.4250) + 0.83×0.3781 = **≈ 0.400** (peor que Morgan solo).
  - Morgan solo: **≈ 0.416**.
  - Comprobación cruzada: el público 0.146 es consistente con
    0.17 × 0.859 ≈ 0.146 (clase-1 ≈ 17%, baseline-1 ≈ 0.86 en clase-1).
- **Umbral:** con ≥ 0.99 el falso enrutamiento en nuevas baja a 0.81%,
  pero la captura de clase-1 cae a 3.72% (spec-split val). El umbral óptimo
  depende de cómo esté construido el oculto; no se calibra aquí.
- **Decisión:** Elán decide si se envía Morgan solo (0.416 esperado),
  blend (0.462 en el escenario favorable, 0.400 en el adverso), o se deja
  el 0.146. Este informe no decide.

## 8. Archivos de esta fase

**Scripts nuevos** (BASE = `~/workspace/enveda-casmi-2026`, logs en
`verificacion/`):
- `12_masas.py` — filtro de masa: índices y precursores por consulta.
- `13_evaluar_masa.py` — retrieval con filtro de masa, 8 variantes, mol/spec.
- `14_entrenar_lineal_binario.py` — piloto WPC Idea 1 (binario + control).
- `15_evaluar_pred.py` — evaluación con filtro de masa de un P arbitrario.
- `16_cobertura_val.py` — top-1 coseno verdadero en val (blend).
- `17_generar_submission_morgan.py` — submission Morgan.
- `validar_huellas_rdkit.py` — equivalencia RDKit 2024.03.6 vs 2026.9.1.

**Intermedios nuevos:** `precursores_full_idx.npz`,
`query_prec_val_mol.npz`, `query_prec_val_spec.npz`,
`query_prec_test.npz`, `P_val_ctrl20k.npy`, `P_val_bin005.npy`,
`P_val_bin010.npy`, `P_val_ctrl20k_full.npy`, `P_val_bin005_full.npy`
(+ `P_val_mol_*.npy`), `W_lineal_ctrl20k.npy`, `W_lineal_bin005.npy`,
`W_lineal_bin010.npy`, `W_lineal_ctrl20k_full.npy`,
`W_lineal_bin005_full.npy`, `bins_piloto_*.npy`, `P_test_lineal.npy`,
`P_test_mol_ids.npy`, `top_idx_test_morgan.npy`, `n_top_test_morgan.npy`,
`muestra_smiles_validacion_rdkit.txt`, `muestra_ik14_validacion_rdkit.csv`.

**Verificación:** `masas_stats.json`, `metricas_masa_val_mol.json`,
`metricas_masa_val_spec.json`, `metricas_masa_*_val_mol.json` (10
evaluaciones piloto/completas), `entrenar_lineal_*.json` (5),
`cobertura_val.json`, `top1cos_val_mol.npy`, `top1cos_val_spec.npy`,
`submission_morgan.json`, `validacion_rdkit.json`,
`huellas_muestra_rdkit2024.03.6.npz`,
`huellas_muestra_rdkit2026.9.1.npz`, `ranks_masa_*.npy`,
`top_idx_masa_*.npy`, `pip_instalacion_sistema.log` y todos los logs de
corrida `12_masas.log`, `13_evaluar_masa_*.log`, `14_entrenar_lineal_*.log`,
`15_evaluar_*.log`, `16_cobertura_val.log`, `17_generar_submission_morgan.log`,
`validacion_rdkit.log` (los tres primeros se movieron desde `logs/` al
cerrar la fase; `logs/` ya no existe).

**Entregable:** `salida/submission_morgan.csv` (md5
`854584724275387743e744fd77757961`).

## 9. Pendientes (no decididos aquí)

1. **Decisión de Elán:** enviar Morgan solo, blend, o dejar el 0.146.
2. **Notebook en runtime** con el modelo elegido (obligatorio para enviar;
   el estático falla en el re-run). Artefactos: W (410 MB) + huellas del
   pool + código de bineado/retrieval.
3. **Ideas WPC 2–5** (bandas m/z, barrido τ × K, re-ranking Jaccard,
   resúmenes compactos) — no intentadas.
4. **Corregir la normalización 05/06 a L2 verdadera** y re-medir el MRR
   del lineal (efecto desconocido).
5. **Lineal binarizado K=50k** sin momento (cabe en RAM, ~1.06 GB) —
   ganancia esperada ~+0.004 MRR, no probado.
6. **kNN con k=25 / pesos coseno²** (variante barata no corrida).
7. **Calibración del umbral de blend** (0.95 vs 0.99) contra el escenario
   real del oculto — no calibrable con datos locales.
