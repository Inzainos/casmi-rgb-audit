# CASMI 2026 — Experimento RGB (espectro → huella Morgan, 3 canales)

Reproducción auditada del experimento RGB del equipo Fractal Core Research
para la competencia Kaggle **Enveda CASMI 2026 — Molecule ID From Mass
Spectra** (métrica MRR@25, envíos solo por notebook, sin internet).

Este repo contiene el código, los resultados de validación, los reportes de
diagnóstico y los submissions locales de referencia del experimento RGB.
**No contiene** datos de la competencia, ni el modelo entrenado (W), ni
credenciales: todo lo grande o secreto vive fuera (ver §5).

## 1. El experimento

El espectro de cada molécula se binea (0.01 Da, top-100 picos por
intensidad) y se transforma en 3 canales por bin m/z — raíz cuadrada de la
intensidad, log1p(intensidad) y presencia binaria — cada canal normalizado
por fila. La matriz resultante (3 × 20,000 columnas) se multiplica por una
matriz lineal `W` (60,000 × 2,048) para predecir la huella de Morgan de
2,048 bits; las predicciones de los espectros de una molécula se promedian.
Se aplica la variante `lineal_bin` (clip [0,1] + top-100 bits a 1) y se
recuperan los 25 mejores SMILES del pool de train por similitud de
Tanimoto, con filtro de masa de precursor (±10 ppm, piso 0.002 Da).

- `codigo/20_rgb.py` — construye la matriz de entrenamiento RGB
  (3 canales por bin m/z) desde la librería bineada.
- `codigo/21_entrenar_lineal_rgb.py` — entrena el modelo lineal
  (K=20k/canal, 6 épocas, batch 1024, lr 0.5, lam 0.0001, momentum 0.9,
  semilla 42).
- `codigo/23_predecir_test_rgb.py` — genera `submission_rgb.csv` para el
  test que reciba (local: test visible; en Kaggle: test oculto).
- `codigo/build_notebook_rgb.py` — construye el notebook de runtime.
- `codigo/casmi_runtime_notebook_rgb.ipynb` — notebook autocontenido que
  genera las predicciones en runtime (el enviado a Kaggle).

## 2. Números (medidos, no supuestos)

Validación mol-split, semilla 42, n=13,748 moléculas
(`validacion/metricas_masa_rgb20k_bin_val_mol.json`):

| Variante | MRR@25 | top-1 | top-5 | recall@25 | mediana rango |
|---|---|---|---|---|---|
| RGB 3 canales, crudo | 0.4542 | 0.3267 | 0.6082 | 0.8301 | — |
| **RGB 3 canales, bin** | **0.4951** | **0.3713** | **0.6458** | **0.8506** | **2** |
| Lineal bin (referencia, 50k bins) | 0.4146 | — | — | — | — |

El RGB bin gana por +0.0805 sobre la referencia lineal_bin (0.4146).

En el test visible (400 moléculas, todas en la librería):

| Archivo | MRR@25 | top-1 | recall@25 |
|---|---|---|---|
| `salida/submission_rgb.csv` (local) | 0.5758 | 0.4550 | 0.9175 |
| Salida del kernel v6 en Kaggle | 0.5723 | 0.4500 | 0.9200 |

En el test oculto (envío 4, ref 58256941, notebook v6, 2026-11-11):

| Envío | Público MRR@25 |
|---|---|
| 3 — baseline 1 (búsqueda en librería) | **0.146** |
| 4 — RGB bin | **0.049** |

## 3. El hallazgo: la validación local mide retrieval con la verdad en el pool

El pool de retrieval son **TODAS las 275,810 moléculas** de train
(`huellas_pool.npz`, 275,810 × 2,048). Las consultas de validación están
**13,790/13,790 dentro del pool** (y spec 9,943/9,943). La evaluación lee
la verdad desde el pool, así que el 0.4951 local mide "¿el modelo ordena
bien la huella verdadera cuando está en el pool?".

En el test oculto solo la fracción clase-1 (~17%, f_class1 ≈ 0.168) tiene
su verdad en el pool. Un modelo de retrieval solo puede devolver moléculas
del pool, así que puntúa **0 en el ~83% de moléculas novedosas por
construcción**. Público esperado del RGB: 0.168 × 0.4951 ≈ 0.083
(0.168 × 0.5758 ≈ 0.097 con el test visible). Observado: 0.049 — la
clase-1 oculta rinde ~0.29 contra ~0.50–0.58 local (residuo real, ver el
reporte formal).

**Consecuencia: un modelo de retrieval de novo tiene un techo duro de
~0.08–0.10 en este test oculto.** Para puntuar en moléculas novedosas
hay que GENERAR estructuras nuevas, no recuperar conocidas. El MRR
mol-split local es la métrica equivocada para ordenar modelos de cara al
test oculto: hay que evaluar con el pool SIN las moléculas de consulta, o
reportar MRR clase-1 × f_class1.

El detalle completo (tres hipótesis probadas con medición, comparación
Kaggle-vs-local, cotas, limitaciones) está en
`diagnostico/DIAGNOSTICO_RGB_0.049_REPORTE.md` (reporte formal) y en
`diagnostico/DIAGNOSTICO_BRECHA_RGB.md` (diagnosis corta).

## 4. Reproducción (1a verificación, 2026-11-11)

Reproducido desde los archivos en esta sesión, con logs en
`validacion/` (este repo) y en el directorio de trabajo del equipo:

- Notebook re-corrido localmente (nbconvert, mismo entorno): salida
  **byte-idéntica** a `salida/submission_rgb.csv`
  (md5 `6dcceab5d57a00b82d485439ebd6061b`, 396,496 B).
- Puntajes en test visible reproducidos exactos: local **0.5758**,
  workspace Kaggle **0.5723** (diff 0.0000 en ambos).
- Composición del pool reproducida: 275,810 moléculas; val 13,790/13,790
  en pool; spec 9,943/9,943 en pool; pool == mols exacto.
- Re-entrenamiento completo desde cero (reconstrucción + entrenamiento +
  evaluación en el sandbox), terminado 2026-11-11 ~21:44 y verificado
  2026-11-12 ~13:24 (logs y JSONs en `validacion/recreacion/`):
  reconstrucción (20) 34.7 s con las 2 verificaciones OK; entrenamiento
  (21) 2,338.5 s, 6/6 épocas, RSSmáx 2,824 MB; evaluación (15) 5.3 s por
  variante. **MSE por época idéntica a la original** (los 6 valores,
  15 dígitos) y **W recreada bit-idéntica a la original**
  (md5 `7a1c0347eb1d7e3a9734791c065c996b`). **MRR@25 reproducido exacto:
  bin 0.4951** (top-1 0.3713, top-5 0.6458, recall@25 0.8506),
  **crudo 0.4542** (top-1 0.3267, top-5 0.6082, recall@25 0.8301),
  n=13,748, cand mediana 106. Incidencia: el agente de la recreación fue
  interrumpido ~21:46 (2 min después de terminar el entrenamiento) antes
  de la evaluación, y su script vigilante tenía un bug de rutas
  (absolutas `/verificacion` en vez de las del sandbox) que hizo fallar
  la evaluación; la evaluación se corrió directamente (exit 0). Los
  artefactos originales quedaron intactos (W 7a1c0347…, bins_rgb
  8413a4e0…). La 2a verificación independiente corre sobre esta
  recreación.

## 5. Qué NO hay en este repo (y dónde está)

- **Datos de la competencia** (`train.parquet` 2.9 GB, `test.parquet`,
  `sample_submission.csv`): archivos oficiales de Kaggle, re-descargables
  desde la competencia
  (https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data).
- **El modelo entrenado** (`W_lineal_rgb20k.npy`, 491,520,128 B) y el
  pool (`huellas_pool.npz` y resto del staging): en el dataset privado de
  Kaggle **elnzainoscorona/rgb-submission-enveda-casmi-2026**, versión 2
  (6 archivos, 555,329,300 B, status Ready):
  https://www.kaggle.com/datasets/elnzainoscorona/rgb-submission-enveda-casmi-2026
- **Credenciales**: ninguna, en ningún archivo (ni token de Kaggle, ni
  claves, ni datos personales). El notebook y los scripts solo leen
  archivos locales; nada llama a ninguna API con credenciales.
- **La librería bineada** (`val_lib.npz`, `full_lib.npz`, ~1.2 GB): se
  reconstruye con los scripts del proyecto base (no incluidos aquí; el
  experimento RGB solo necesita la librería ya construida para
  `20_rgb.py`).

## 6. Cómo auditarlo

1. Verificar los md5 de `salida/` contra la tabla de §7 y los de
   `validacion/` contra los JSONs citados.
2. Re-correr `codigo/casmi_runtime_notebook_rgb.ipynb` (nbconvert) con
   `test.parquet` y el dataset V2 en `/kaggle/input` (o con los fallbacks
   locales `datos/` y `runtime_rgb_dataset/`): la salida debe ser
   byte-idéntica a `salida/submission_rgb.csv`.
3. Re-correr la validación: `20_rgb.py --k-cols 20000`,
   `21_entrenar_lineal_rgb.py --tag rgb20k --epochs 6 --momentum 0.9`,
   `15_evaluar_pred.py` (no incluido aquí; vive en el proyecto base) con
   `--tag rgb20k_bin --split mol --bin` sobre las predicciones: debe dar
   MRR@25 0.4951 (n=13,748).
4. Reproducir la composición del pool: el pool tiene 275,810 moléculas
   (todas las de train) y las consultas val están 13,790/13,790 dentro.

## 7. Submissions locales (md5)

| Archivo | md5 | Notas |
|---|---|---|
| `salida/submission_rgb.csv` | `6dcceab5d57a00b82d485439ebd6061b` | RGB bin; enviado como envío 4 (público 0.049) |
| `salida/submission.csv` | `7a7da389417fb162039a94394d76e1ee` | Baseline 1; envío 3 (público 0.146) |
| `salida/submission_relleno25.csv` | `7f082916a4a2bd68392750ea3cf2c8a7` | Baseline 1 rellenado a 25 |
| `salida/submission_morgan.csv` | `854584724275387743e744fd77757961` | Lineal Morgan (0.4146); no enviado |
| `salida/submission_paso2.csv` | `ed758b99cdb3a91ac690f02fc3618779` | Paso 2 kNN; no enviado |

Fechas de la competencia: entrada y fusión de equipos 7 dic 2026;
envío final 14 dic 2026 (11:59 PM UTC). Selección final: 0/2.

## 8. Piloto B: generación huella → SMILES (experimental)

Se agregó el decodificador Transformer del piloto B en `codigo/piloto_b/`,
junto con el notebook Colab y las métricas agregadas en `validacion/piloto_b/`.
No sustituye RGB: su MRR@25 de 0.3372 se midió en 2,000 GEN-VAL con huellas
exactas, no con espectros ni con huellas predichas. El exact-match greedy
del holdout fue 0.232 (puerta propuesta: 0.50). Se ejecutó además la
integración RGB→SMILES en 2,000 consultas GEN-VAL (mismo muestreo que la
prueba con huellas exactas): MRR@25 **0.0005** (IC95% 0–0.0015), con solo
1 verdad recuperada en top-25 y Tanimoto top-1 medio 0.1541.

La novedad se midió contra el conjunto exacto de 50,000 moléculas usado por
el checkpoint probado (el plan contempla escalar a 252,077). MRR@25 novel
es **0.000500** por identidad directa (n=2,000) y **0.000502** con
canonicalización tautomerizada del scorer (n=1,994; 6 consultas excluidas
por solapamiento). Ambas cifras están bajo el umbral propuesto de 0.01.
Aunque el 99.71% de los candidatos válidos distintos no está en el train del
decodificador, eso no implica cercanía a la verdad: el Tanimoto top-1 sigue
en 0.1541. El resultado solo corresponde a GEN-VAL, no a moléculas ocultas
novedosas. Por tanto, B **no está listo para envío** ni reemplaza RGB.

Se hizo un primer fine-tune con huellas RGB out-of-fold: el modelo RGB se
entrenó sin las 50,000 moléculas de B, y para el ajuste se mezcló 50% de
huellas OOF con las huellas exactas y corrupción sintética existentes. En la
misma muestra GEN-VAL (n=2,000), `MRR_novel` tautomerizado subió a **0.001504**
(recall@25 0.007523; 15 verdades en top-25), validez a **0.9080** y
Tanimoto top-1 a **0.2168**. El delta de MRR pareado fue +0.001002, pero su
IC95% bootstrap incluye cero (−0.000174–0.002085); es una mejora puntual
prometedora, no concluyente, y aún queda lejos del umbral 0.01.
Los detalles y límites están en
[`diagnostico/RESULTADO_PILOTO_B.md`](diagnostico/RESULTADO_PILOTO_B.md).

La revisión y los pasos pendientes están en
[`diagnostico/RESULTADO_PILOTO_B.md`](diagnostico/RESULTADO_PILOTO_B.md).
Se añadió una ruta de evaluación RGB→SMILES con `--predicciones-rgb`, que
aplica la misma binarización `lineal_bin` (top-100) y alinea por
`molecule_id`; requiere las predicciones RGB de validación que no están en
este repositorio. El entrenamiento acepta además `--predicciones-oof` para
mezclar errores RGB reales durante el fine-tune desde `--checkpoint-inicial`.
El paquete de datos, el modelo entrenado y las predicciones por molécula
siguen fuera del repositorio.
