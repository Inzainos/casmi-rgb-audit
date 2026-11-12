# RESULTADOS — CASMI sandbox Kali, pruebas del espectrograma

**Fecha:** 2026-11-10 (trabajo 14:00–15:45 CST aprox.)
**Orden:** Elán, 2026-11-10 ~13:54, vía Coordinador. Elán hizo `pruebas/` para
que el chat de CASMI suba lo necesario y corra ahí las pruebas del
espectrograma (en el sandbox cuántico no se puede); él no lo sube ni lo corre.
**Reglas:** nada a Kaggle, nada a main/GitHub sin decisión de Elán. No hay
commits en el repo del sandbox. Nada fuera de `pruebas/` en la Kali.
**Destino:** `/home/deamon/wt_deamonx/CASMI-sandbox/pruebas/`
(worktree de `/home/deamon/CASMI`, vacío salvo README + .git).
**Origen:** `~/workspace/enveda-casmi-2026/` (repo CASMI, esta caja).
**Copia local de este informe:**
`~/workspace/enveda-casmi-2026/verificacion/RESULTADOS.md` (mismo contenido).

## 0. Resumen ejecutivo

- **Subida completa y verificada:** 323 archivos (4,445,259,032 B) en su
  layout correcto. Verificación final: **302/323 md5 exactos**; los 21
  restantes son los .py con el parche BASE/14, verificados por `diff`
  (la única diferencia es el parche, ver §9).
- **Tres pruebas corridas, las tres medidas con MRR@25 mol-split
  (n=13,748, filtro de masa ±10 ppm, semilla 42). Ninguna gana** contra el
  mejor modelo Morgan, lineal_bin K=50k (**0.4146**):
  1. Lineal binarizado K=50k sin momento: crudo **0.2582**, bin **0.3036** —
     NO GANA (−0.111 vs 0.4146). La proyección +1% del informe Morgan queda
     refutada para esta receta.
  2. WPC Idea 2, conteos por banda m/z (solo bandas): crudo **0.1015**, bin
     **0.1069** — señal real pero marginal (nulo: 0.0981). No se corrió el
     combinado bandas+espectro (ver §6).
  3. kNN k=25 / pesos coseno²: crudo **0.1517**, bin **0.1366** — mejora
     +0.011 sobre el kNN original con filtro (0.1406) pero sigue lejos del
     lineal. NO GANA.
- **El estado de la entrega no cambia:** el mejor modelo de novo sigue
  siendo lineal_bin K=50k (0.4146 mol-split); el envío 3 público sigue en
  0.146. El blend Morgan + baseline 1 sigue pendiente de decisión de Elán.

## 1. Qué se subió (inventario verificado 2026-11-10 ~15:40 CST)

| Directorio | Archivos | Bytes |
|---|---|---|
| raíz (`*.py`, `*.md`, `.gitignore`, notebook) | 35 | ~281 KB |
| `intermedios/` | 83 | 4,375,861,454 |
| `verificacion/` (incl. `runtime_check/`) | 201 | 67,840,308 |
| `salida/` (4 submissions) | 4 | 1,557,270 |
| **Total** | **323** | **4,445,259,032** |

- **NO se subió `datos/`** (train.parquet 2.9 GB, SHA-256
  `9423f90ec08b941a9646b535bf0606183e86ff0c5a297978f5a70693d41bcd98`):
  las pruebas usan los datos ya preparados. Si una prueba exige
  reconstruir desde el parquet crudo, avisar antes de subir 3 GB.
- Manifiesto correcto (md5 + tamaño + ruta relativa, 323 filas):
  `pruebas/local_manifiesto.tsv`. **OJO:** `pruebas/md5sumas-origen.txt`
  (generado 14:15) tiene formato defectuoso — columnas/rutas
  inconsistentes, 321 líneas con solo basename y sin directorios — y NO
  sirve para verificar con `md5sum -c`. Queda como registro histórico;
  la verificación se hizo contra el manifiesto correcto.
- Corrección al LEEME: la raíz tiene **35** archivos, no 41 (el LEEME
  decía 41; el conteo verificado del manifiesto es 35).

### md5 clave verificados en destino

```
34c36a2c9b1a7b2e2cf5df8750ba21b3  intermedios/X_train_paso2.npz
e7a74ad39ce908ae03a4185b2b983627  intermedios/y_train_paso2.npz
65d0fb24298c9473772a4cbc8287ce16  intermedios/bins_sel_paso2.npy
850ddda297f15f63f42785edd96071bc  intermedios/val_qmol.npz
613d5b2abf75f602bc9c7c2034d32099  intermedios/val_qmol_meta.npz
0d78e46583807a89f8372330b522b0b8  intermedios/huellas_pool.npz
865537e2697f2ce4fad7bf2ac6ca6f04  intermedios/precursores_full_idx.npz
fd3a6e4e063a3f62bcdf87d76a2b0f3d  intermedios/query_prec_val_mol.npz
e9c5a3cf77622b4cfdfb44b6f450a3d8  intermedios/val_lib.npz (546,689,919 B)
a896f80cf764fdb4a79004ad05d9833b  intermedios/val_lib_meta.npz (22,863,724 B)
371ed2a1133486197b62f91b28c6326e  intermedios/W_lineal_paso2.npy (409,600,128 B)
65e022e8a0ebc8442ffb1eb7711a162f  intermedios/full_lib.npz (584,190,941 B)
22a6508151d6fd32ba7c7793389b0c71  intermedios/full_lib_meta.npz
7a7da389417fb162039a94394d76e1ee  salida/submission.csv
ed758b99cdb3a91ac690f02fc3618779  salida/submission_paso2.csv
7f082916a4a2bd68392750ea3cf2c8a7  salida/submission_relleno25.csv
854584724275387743e744fd77757961  salida/submission_morgan.csv
7f082916a4a2bd68392750ea3cf2c8a7  verificacion/runtime_check/submission.csv
                                 (idéntico contenido a submission_relleno25.csv)
```

Los 4 submissions locales en `salida/` coinciden byte a byte con los
originales (md5 arriba, iguales a los de AGENTS.md regla 6). Los
originales de esta caja no se tocaron en ningún momento (solo lectura).

## 2. Incidentes durante la transferencia (registro honesto)

1. **Dos reinicios del runtime** mataron rsyncs en curso (SIGTERM, socat)
   y limpiaron `/tmp` en esta caja (se perdieron los scripts de parche y
   la lista md5 local; se recrearon y re-verificaron: el `07b` se
   reescribió desde `07_knn_huella.py` y se verificó por `diff` que los
   cambios son exactamente k=25, pesos cos² y solo mol-split).
2. **El rsync de reanudación aplana con slash final.** La reanudación se
   lanzó como `rsync … intermedios/ verificacion/ salida/ destino/` — con
   slash final rsync copia el *contenido* y aplana todo en `pruebas/`
   raíz. Se detectó por conteos (`intermedios/` 56 archivos,
   `verificacion/` solo logs de la Kali, `salida/` vacía) y por `find`.
   El contenido estaba completo (234 archivos en la última pasada), solo
   desordenado y con duplicados.
3. **Reparación** con `pruebas/reparar_layout.py` (script guardado en la
   Kali, manifiesto `local_manifiesto.tsv`): compara md5+tamaño y mueve
   cada archivo a su ruta correcta.
   - 243 movidos de la raíz plana a su subdirectorio.
   - 1 sobrescrito: `intermedios/P_val_topo_v2.npy` — la copia en
     `intermedios/` estaba truncada (restos de la corrida original muerta
     por SIGTERM); la copia plana verificaba bien por md5 y se movió encima.
   - 44 duplicados planos borrados (el destino ya verificaba bien).
   - **2.42 GB liberados** (pruebas/ pasó de 7,653,878,849 B a
     5,229,020,392 B).
   - Re-verificación posterior independiente: 302 OK, 0 por mover,
     0 duplicados, 0 faltantes; solo los 21 .py parcheados difieren
     (por diseño, ver §9).
4. **Lección:** con rsync `--partial`, el progreso real va en el temporal
   oculto; el `ls` del archivo destino no se mueve hasta el renombre
   final — un `ls` sin cambio no significa que la transferencia esté parada.
5. **Subida de 2 archivos que faltaban** (no son .py/.md ni están en los
   tres directorios, por eso ningún rsync los llevó): `.gitignore` (58 B,
   md5 `9da3214c…`) y `casmi_runtime_notebook.ipynb` (19,703 B, md5
   `a7e64ac8…`), ambos verificados por md5 tras subirlos. Los 3
   `__pycache__/*.pyc` locales (bytecode Python 3.12/3.13 de esta caja)
   quedaron fuera del manifiesto a propósito; la Kali genera su propio
   `__pycache__` (3.14).

## 3. Entorno final en la Kali

- **Máquina:** 16 CPU, 7 GB RAM total (~5 GB disponibles), 749 GB disco libre.
- **Venv:** `/home/deamon/.venv-casmi` (Python **3.14.7**,
  `--system-site-packages`).
- **Versiones:** numpy **2.4.6**, scipy **1.17.1**, pandas **2.3.3**,
  scikit-learn **1.9.0**, pyarrow **26.0.1** (del sistema) + **RDKit
  2026.9.1** (rueda cp314 instalada por pip, 31.8 MB).
- El pipeline se construyó y midió en esta caja con Python 3.12.3 +
  numpy 1.26.4 (nunca 2.x) + scipy 1.11.4 + RDKit 2024.03.6. **RDKit
  2026.9.1 ya estaba validado por el informe Morgan como idéntico a
  2024.03.6 (0 bits distintos en la muestra de 500 SMILES).**
- **Compatibilidad numpy 2.x — VERIFICADA, sin bloqueos.** Prueba corta
  2026-11-10 ~14:17 (log `pruebas/verificacion/` — ver §8): imports de
  `casmi_lib`, 13 y 14 OK; carga de X_train (261,536 × 50,000, nnz
  22,072,956, float32), y_train (261,536 × 2,048, nnz 12,045,549, bool),
  bins_sel (50,000), val_qmol (38,377 × 200,000, nnz 2,417,000),
  huellas_pool (275,810 × 2,048, nnz 12,678,178),
  precursores_full_idx (2,438,064), query_prec_val_mol (mol 13,748;
  prec 25,431); `transformar()` bin tau=0.05 sobre 2,000 filas val OK
  (nnz 22,546, media 0.2766, máx 1.0); Morgan de cafeína con RDKit = 25
  bits. No se encontró ningún nombre eliminado de numpy 2.0 en los
  scripts (grep previo). No hizo falta crear venv compatible alternativo.
- **Enlace:** ~1.0–1.5 MB/s compartido (rsync). La subida completa tomó
  varias pasadas por los reinicios; la última pasada envió 2,791,219,676 B
  en ~45 min.

## 4. Prueba 1 — Lineal binarizado K=50k sin momento (PRIMERA PRUEBA)

Variante no medida del informe Morgan §3.2 / §next steps (~+1% proyectado
sobre 0.4146). Dos variables a la vez: binarización + sin momento.

- **Comando (Kali, PID 2029990):**
  `~/.venv-casmi/bin/python 14_entrenar_lineal_binario.py --tag bin050k_full --binario --tau 0.05 --k-cols 50000 --subset 0 --epochs 6 --batch 1024 --lr 0.5 --lam 1e-4 --momentum 0 --seed 42`
- **Entrenamiento:** 1,021.4 s, RSS máx 1,104 MB. Binarización:
  nnz 22,072,956 → 4,066,587 (18.4% de picos con tau=0.05). MSE por época:
  0.01959 → 0.01816 → 0.01769 → 0.01740 → 0.01719 → **0.01703**,
  |W|_F 43.3. El MSE seguía bajando en la época 6 (subentrenado).
- **Evaluación** con 15, mol-split, filtro de masa, n=13,748:
  - Crudo: **MRR@25 0.2582**, Top-1 0.1520, Top-5 0.3724, Recall@25 0.6685,
    Tanimoto 0.2030.
  - Bin (clip [0,1] + top-100): **MRR@25 0.3036**, Top-1 0.1932,
    Top-5 0.4280, Recall@25 0.7098, Tanimoto 0.2253.
- **Veredicto: NO GANA.** −0.111 absoluto (−26.8% relativo) vs lineal_bin
  K=50k 0.4146; incluso por debajo del piloto bin005 K=20k (0.3251, 80k
  filas, 3 épocas). Causa más probable: subentrenamiento sin momento
  (MSE final 0.01703 vs 0.01569 de bin005_full, y seguía bajando), no la
  binarización en sí — como cambiaron dos variables a la vez, la pérdida
  NO puede atribuirse solo a la binarización. La proyección +1% queda
  refutada para esta receta.
- **Salidas (Kali):**
  `pruebas/intermedios/W_lineal_bin050k_full.npy` (50,000 × 2,048, 410 MB),
  `pruebas/intermedios/P_val_bin050k_full.npy`,
  `pruebas/intermedios/P_val_mol_bin050k_full.npy`,
  `pruebas/intermedios/bins_piloto_bin050k_full.npy`,
  `pruebas/verificacion/entrenar_lineal_bin050k_full.json`,
  `pruebas/verificacion/14_entrenar_lineal_bin050k_full.log`,
  `pruebas/verificacion/15_evaluar_bin050k_full_val_mol.log`,
  `pruebas/verificacion/15_evaluar_bin050k_full_bin_val_mol.log`.

## 5. Prueba 2 — WPC Idea 2: conteos por banda m/z

- **Método:** script `18_bandas.py` (nuevo, en `pruebas/`): por espectro,
  conteos de picos presentes en 20 bandas m/z de 100 Da (0–2000 Da) +
  total de picos + fracción ocupada → 22 features.
  `19_entrenar_lineal_bandas.py` (nuevo): lineal solo-bandas (K=22, SGD
  lr 0.5, lam 1e-4, mom 0.9, 6 épocas, L2 por fila; 38.8 s; MSE 0.01736;
  W 180 KB). Medias de banda train: [17.7, 41.9, 14.6, 8.3, 1.0, 0.4,
  0.2, 0.1, 0, …]; picos totales media 84.4 train / 63.0 val.
- **Evaluación** con 15, mol-split, filtro de masa, n=13,748:
  - Crudo: **MRR@25 0.1015**, Top-1 0.0452, Recall@25 0.4006,
    Tanimoto 0.2307.
  - Bin: **MRR@25 0.1069**, Top-1 0.0493, Recall@25 0.4102.
- **Veredicto: señal real pero marginal** — apenas sobre el nulo (0.0981):
  +0.0034 / +0.0088. Tanimoto 0.2307 supera al del lineal, y aun así el
  MRR es pobre (misma lección que en paso 2: Tanimoto no predice MRR).
- **Combinado bandas+espectro NO corrido a propósito:** las bandas son un
  engrosamiento determinista del espectro que el lineal ya ve; ganancia
  esperada ≈ 0.
- **Asimetría documentada:** las bandas train vienen de las top-50k
  columnas y las val de las 200k columnas, así que las bandas train ≥9
  son ~0 mientras las val son pequeñas pero no cero; además las filas
  train son espectros max-nnz y las val son todos los espectros.
- **Salidas (Kali):**
  `pruebas/intermedios/B_train_paso2.npy` (261,536 × 22),
  `pruebas/intermedios/B_val_qmol.npy` (38,377 × 22),
  `pruebas/intermedios/W_lineal_bandas.npy`,
  `pruebas/intermedios/P_val_bandas.npy`,
  `pruebas/intermedios/P_val_mol_bandas.npy`,
  `pruebas/verificacion/18_bandas.log`,
  `pruebas/verificacion/bandas_stats.json`,
  `pruebas/verificacion/19_entrenar_lineal_bandas.log`,
  `pruebas/verificacion/entrenar_lineal_bandas.json`,
  `pruebas/verificacion/15_evaluar_bandas_val_mol.log`,
  `pruebas/verificacion/15_evaluar_bandas_bin_val_mol.log`.

## 6. Prueba 3 — kNN k=25 / pesos coseno²

- **Método:** `07b_knn_k25_cos2.py` (nuevo, en `pruebas/`, derivado de
  `07_knn_huella.py`): TOPK_VECINOS 25 (era 10), pesos =
  clip(coseno, 0)² normalizado (era coseno lineal), solo mol-split.
  Reescrito tras el reinicio y verificado por `diff` contra el 07
  original (cambios exactamente esos tres).
- **Corrida:** 56.4 s, RSS máx 2,462 MB. Librería val_lib: 2,285,808
  filas, 117,531,907 nnz, prec 100.0–1986.1. 38,377 espectros, 528 sin
  candidatos, top1 cos media 0.547, mediana 0.546, ≥0.9: 17.0%.
  Pred media 0.0216, máx 1.000, 13,748 moléculas.
- **Evaluación** con 15, mol-split, filtro de masa, n=13,748:
  - Crudo: **MRR@25 0.1517**, Top-1 0.0215, Top-5 0.3011, Recall@25 0.6045,
    Tanimoto 0.3209.
  - Bin: **MRR@25 0.1366**, Top-1 0.0201, Top-5 0.2811, Recall@25 0.6076,
    Tanimoto 0.2197.
- **Veredicto: NO GANA.** Mejora +0.0111 (+7.9% relativo) sobre el kNN
  original con filtro (0.1406), pero sigue lejísimos del lineal_bin
  K=50k (0.4146); la vía kNN sigue sin ser competitiva. El bin aquí
  empeora (al revés que en el lineal). Tanimoto 0.3209 es el más alto
  entre variantes kNN y aun así el MRR es pobre: Tanimoto no predice MRR.
- **Salidas (Kali):**
  `pruebas/intermedios/P_val_knn25_cos2.npy` (112,623,744 B),
  `pruebas/intermedios/P_val_mol_knn25_cos2.npy`,
  `pruebas/verificacion/knn25_cos2.json`,
  `pruebas/verificacion/07b_knn_k25_cos2.log`,
  `pruebas/verificacion/15_evaluar_knn25_cos2_val_mol.log`,
  `pruebas/verificacion/15_evaluar_knn25_cos2_bin_val_mol.log`.

## 7. Tabla resumen — MRR@25 mol-split, filtro de masa (n=13,748)

| Modelo | MRR@25 | Top-1 | Top-5 | Recall@25 | Tanimoto |
|---|---|---|---|---|---|
| **lineal_bin K=50k (mejor Morgan)** | **0.4146** | — | — | — | — |
| topo_v2 | 0.4123 | — | — | — | — |
| bin005_full (K=20k) | 0.4023 | — | — | — | — |
| ctrl20k_full | 0.3985 | — | — | — | — |
| lineal crudo (W_lineal_paso2) | 0.3719 | — | — | — | — |
| **bin050k_full crudo (prueba 1, Kali)** | **0.2582** | 0.1520 | 0.3724 | 0.6685 | 0.2030 |
| **bin050k_full bin (prueba 1, Kali)** | **0.3036** | 0.1932 | 0.4280 | 0.7098 | 0.2253 |
| **kNN k=25/cos² crudo (prueba 3, Kali)** | **0.1517** | 0.0215 | 0.3011 | 0.6045 | 0.3209 |
| kNN original con filtro (k=10, cos) | 0.1406 | — | — | — | — |
| **kNN k=25/cos² bin (prueba 3, Kali)** | **0.1366** | 0.0201 | 0.2811 | 0.6076 | 0.2197 |
| **bandas solo crudo (prueba 2, Kali)** | **0.1015** | 0.0452 | — | 0.4006 | 0.2307 |
| **bandas solo bin (prueba 2, Kali)** | **0.1069** | 0.0493 | — | 0.4102 | — |
| nulo | 0.0981 | — | — | — | — |
| kNN sin filtro (paso 2) | 0.0346 | — | — | — | 0.3148 |
| oráculo | 0.9937 | — | — | — | — |

Referencia externa: envío 3 público (notebook en runtime) MRR@25 = **0.146**.

## 8. Salidas generadas en la Kali (rutas exactas)

Conteo final en `pruebas/`: `intermedios/` 94 archivos (83 subidos +
11 generados en la Kali), `verificacion/` 233 archivos (201 subidos + 32
generados), `salida/` 4 archivos, raíz 39 `.py/.md/.ipynb` (35 subidos +
`07b_knn_k25_cos2.py`, `18_bandas.py`, `19_entrenar_lineal_bandas.py`,
`reparar_layout.py`). Total `pruebas/`: 5,229,020,392 B.

- Prueba corta compatibilidad: log en
  `pruebas/verificacion/` (corrida 2026-11-10 ~14:17; ver §3).
- Prueba 1: ver §4. Prueba 2: ver §5. Prueba 3: ver §6.
- Métricas JSON de cada evaluación 15 en
  `pruebas/verificacion/metricas_masa_<tag>_val_mol.json` (+
  `ranks_masa_<tag>_val_mol.npy`, `top_idx_masa_<tag>_val_mol.npy`).

## 9. Parches y scripts nuevos (SOLO copias de la Kali; originales intactos)

1. **BASE** (21 archivos .py): `BASE = Path("/home/hatch/...")` →
   `BASE = Path("/home/deamon/wt_deamonx/CASMI-sandbox/pruebas")`, con
   comentario `# PARCHE SANDBOX KALI 2026-11-10 (era /home/hatch/...)`.
   Verificado por `diff` contra los originales: la única diferencia en
   19 archivos es esa línea; en `explorar.py`/`explorar2.py` es la línea
   `D = "…/datos"` (con nota "datos no subidos"); en 14 además el parche
   siguiente. `casmi_lib.py`, `paso2_lib.py`, `topo_lib.py` no tienen
   BASE y están byte-idénticos a los originales.
2. **14 sin-momento** (`14_entrenar_lineal_binario.py`): con
   `--momentum 0` no se asignan V ni tmp (ahorra 820 MB a K=50k);
   decaimiento desacoplado `W *= (1 - lr*lam)` antes de restar el
   gradiente — matemáticamente idéntico a la rama con momento y mom=0
   (salvo redondeo float32 de último bit). Con momentum != 0 el código
   es byte-idéntico al original. El JSON registra
   `"sin_momento_sin_V_tmp": true`.
3. **Scripts nuevos en `pruebas/`:** `07b_knn_k25_cos2.py` (variante kNN,
   §6), `18_bandas.py` + `19_entrenar_lineal_bandas.py` (Idea 2, §5),
   `reparar_layout.py` + `local_manifiesto.tsv` (reparación del layout,
   §2).
4. Los scripts de parche originales (`parche_base.py`,
   `parche_14_sin_momento.py`) se perdieron con la limpieza de `/tmp` en
   el reinicio; los parches viven aplicados en las copias de la Kali y
   están documentados aquí y en el LEEME.

## 10. Pendiente / no hecho

- **Notebook en runtime de Morgan NO empezado** (preparación de envío;
  necesita decisión de Elán). El archivo
  `casmi_runtime_notebook.ipynb` sí está subido a `pruebas/` (copia de
  trabajo, no es un envío).
- **Blend Morgan + baseline 1** (esperado ≈0.462 favorable / ≈0.400
  adverso en oculto) sigue pendiente de decisión de Elán.
- **Ideas WPC 3–4 intentadas en la Ronda 2** (ver sección Ronda 2 al final):
  barrido τ×K (mejor 0.4035, no gana) y re-ranking Tanimoto (no gana,
  0.3874). Queda la Idea 5 (higiene, no score). Idea 1 ya estaba medida
  (+0.95% sin superar 0.4146).
- **Nada a Kaggle, nada a main.** Envíos: 3 de 5 diarios usados (al
  2026-11-09); selección final 0/2. Fechas: entrada/cierre de equipos
  7 dic 2026; envío final 14 dic 2026 (23:59 UTC).
- `datos/` (train.parquet 3 GB) sigue sin subirse; avisar antes si una
  prueba lo exige.
- El `LEEME_SANDBOX.md` dice 41 archivos en raíz; el conteo verificado
  es 35 (ver §1). Se deja constancia aquí; el LEEME lleva apéndice de
  corrección con fecha 2026-11-10 ~15:45.

---

## Ronda 2 — ideas de la trilogía de física (caps. 35/36/37) aplicadas al proyecto

**Fecha:** 2026-11-10 ~17:02–19:39 CST. **Orden:** Elán, 2026-11-10 ~16:55
("aplícalo al proyecto", vía chat de entropía). Especificación:
`wpc-schumann-ideas-cap37.md` (copiada a `pruebas/` como referencia).
**Reglas:** nada a Kaggle, nada a main/GitHub, ningún commit en el sandbox,
nada fuera de `pruebas/`, no se subió `datos/`. Driver secuencial
`run_ronda2.sh` (un proceso pesado a la vez), log
`verificacion/ronda2_driver.log` (**0 Tracebacks**). Referencia a batir:
**lineal_bin K=50k, MRR@25 mol-split = 0.4146** (n=13,748, semilla 42,
filtro de masa ±10 ppm). Todas las MRR de esta sección salen de los JSON
de `verificacion/metricas_masa_*_val_mol.json` (script 15) y de
`metricas_rerank_lineal_bin_val_mol.json` (script 22, corregido, ver §5).

### Resumen ejecutivo

- **(a) RGB 3 canales GANA — y por mucho: MRR@25 bin 0.4951 (+0.0805 sobre
  0.4146), crudo 0.4542 (+0.0821 sobre el lineal crudo 0.3721).** Nuevo mejor
  modelo de novo. Top-1 0.3713, top-5 0.6458, recall@25 0.8506 (bin).
- **(b) CE como feature SALTADA:** `collision_energy_ev` no existe en ningún
  archivo de `pruebas/` (las metas solo traen precursor_mz, mol,
  ingest_lib). Vive solo en train.parquet (3 GB, no subido). Falta una tabla
  por espectro con la CE, extraíble del parquet en esta caja — no se hizo
  sin preguntar, como ordenaba la tarea.
- **(6) bin050k CON momento: bin 0.4035, crudo 0.3568 — NO GANA** (a 0.0111
  de 0.4146). El momento vale +0.0999 en bin (0.3036 → 0.4035) y +0.0986 en
  crudo (0.2582 → 0.3568): la caída de la ronda 1 era subentrenamiento,
  no la binarización.
- **WPC Idea 3 (barrido τ×K, momento 0.9, 6 épocas, bin):** τ=0.05 es el
  mejor (K=20k 0.4023, K=50k 0.4035); τ=0.10 → 0.3689/0.3696; τ=0.20 →
  0.3279/0.3281. K (20k vs 50k) no mueve el número (±0.001). **Ninguna
  celda supera 0.4146.**
- **WPC Idea 4 (re-ranking Tanimoto): NO GANA.** La huella recomputada con
  RDKit desde los SMILES es idéntica al pool en 137,882/137,882 (el
  re-ranking por Tanimoto-bin es un no-op probado: orden idéntico en
  13,748/13,748). Re-ordenar por Tanimoto crudo EMPEORA: 0.4146 → 0.3874.
- **Referencia re-verificada con el script 15:** lineal_bin 0.414625,
  lineal crudo 0.372073 (la fase Morgan daba 0.414578/0.371937 con el
  script 13; la diferencia es de empates, ~1 consulta).
- **El estado de la entrega cambia:** el mejor modelo de novo ya no es
  lineal_bin (0.4146) sino **RGB 0.4951**. El envío 3 público sigue en
  0.146. Qué se hace con RGB (notebook en runtime, blend) es decisión
  de Elán.

### Tabla de resultados (MRR@25 mol-split, n=13,748, filtro ±10 ppm)

| Prueba | Configuración | Crudo | Bin | Veredicto |
|---|---|---|---|---|
| Ref. lineal (06, K=50k, mom 0.9, 5 ép.) | sqrt+L2 | 0.3721 | **0.4146** | referencia |
| (a) RGB 3 canales (K=20k/canal, mom 0.9, 6 ép.) | sqrt/log/bin | 0.4542 | **0.4951** | **GANA +0.0805** |
| (6) bin050k CON momento (τ=0.05, K=50k, 6 ép.) | binario | 0.3568 | 0.4035 | no gana (−0.0111) |
| (6) bin050k SIN momento, ronda 1 | binario | 0.2582 | 0.3036 | no gana (momento +0.0999) |
| Idea 3 (τ=0.05, K=20k) = bin005_full | binario | — | 0.4023 | no gana |
| Idea 3 (τ=0.10, K=20k) | binario | — | 0.3689 | no gana |
| Idea 3 (τ=0.10, K=50k) | binario | — | 0.3696 | no gana |
| Idea 3 (τ=0.20, K=20k) | binario | — | 0.3279 | no gana |
| Idea 3 (τ=0.20, K=50k) | binario | — | 0.3281 | no gana |
| Idea 4 re-rank Tanimoto crudo | sobre lineal_bin | — | 0.3874 | empeora (−0.0272) |

### 1. (a) RGB 3 canales — GANA

Scripts nuevos `20_rgb.py` / `21_entrenar_lineal_rgb.py` (misma SGD con
momento de 14, misma agregación por molécula de 06).

- **Build (20, 28.7 s, RSSmáx 1,671 MB):** replica exactamente la
  selección de 05 (1 espectro/molécula, top-20k bins por df) y lo verifica:
  train_mol idéntico a `train_mol_paso2.npy` y los 20k bins son subconjunto
  de los 50k de 05 (df mínimo incluido 81). `X_train_rgb.npz`: 261,536 ×
  60,000, nnz 61,745,664 (20,581,888 por canal). Canales, cada uno
  normalizado por fila como en 06 (dividir por sqrt(suma de valores)):
  sqrt, log1p, binario.
- **Train (21, 2,398 s, RSSmáx 2,869 MB):** 6 épocas, lr 0.5, lam 1e-4,
  momento 0.9, semilla 42, subset completo. MSE por época:
  0.01644 → 0.01530 → 0.01501 → 0.01485 → 0.01474 → **0.01466** — el MSE
  final más bajo de todos los lineales (bin005_full 0.01569,
  ctrl20k_full 0.01605, bin050k_mom 0.01572). `W_lineal_rgb20k.npy`:
  60,000 × 2,048, 491,520,128 B. `P_val_rgb20k.npy`: 13,748 × 2,048,
  pred media 0.0182, máx 2.162.
- **Eval (15):** crudo 0.454190 (top-1 0.3267, top-5 0.6082, recall
  0.8301), bin 0.495100 (top-1 0.3713, top-5 0.6458, recall 0.8506),
  n=13,748, candidatos/mol mediana 106.
- **Lectura honesta:** mismo protocolo, semilla y split que la referencia;
  ganan las DOS variantes (crudo y bin), así que no es un artefacto del
  binarizado. 6 épocas vs 5 de la referencia no explican +0.08 (los
  controles a 6 épocas están en ~0.40) y K total 60k vs 50k tampoco (el
  barrido τ×K muestra que K no mueve el número). No hay fuga: mismas filas
  de entrenamiento y mismos Y que 05; las moléculas de validación siguen
  fuera del entrenamiento. Lo que falta para convertirlo en envío es el
  notebook en runtime (W 491 MB + huellas + índice de masa) y la decisión
  de Elán.

### 2. (b) CE como feature — SALTADA (sin dato)

Verificado antes de correr nada: `collision_energy_ev` no está en ningún
archivo de `pruebas/`. Revisados: `val_lib_meta.npz`,
`val_qmol_meta.npz`, `full_lib_meta.npz`, `val_qspec_meta.npz` (solo
precursor_mz, mol, ingest_lib), `query_prec_val_mol.npz` (mol, prec,
indptr), `precursores_full_idx.npz` (prec, mol). La única mención de
"collision" en `pruebas/*.py` es `explorar.py`, que la lee del parquet.
**Lo que falta exactamente:** una tabla por espectro con la CE, alineada
con el orden de filas de `val_lib.npz` / `val_qmol.npz`. Es extraíble de
train.parquet en esta caja (`explorar.py` confirma que la columna existe
ahí, como lista por espectro), pero la tarea ordenaba no subir
train.parquet (3 GB) sin preguntar, así que no se construyó ni se subió.
Próximo paso si Elán lo quiere: construir el extracto pequeño en esta
caja y subir solo ese archivo.

### 3. (6) bin050k CON momento — NO GANA, pero aísla el efecto

`14_entrenar_lineal_binario.py --tag bin050k_mom --binario --tau 0.05
--k-cols 50000 --subset 0 --epochs 6 --momentum 0.9` (misma receta que la
prueba 1 de la ronda 1, solo cambia el momento). MSE por época:
0.01759 → 0.01637 → 0.01605 → 0.01589 → 0.01579 → **0.01572**. Eval (15):
crudo 0.356804, bin 0.403509 (top-1 0.2843, top-5 0.5450, recall 0.7837).
**El momento vale +0.0999 en bin y +0.0986 en crudo** respecto a la ronda 1
sin momento (0.3036 / 0.2582). Con momento, el binario a K=50k (0.4035)
queda igual que a K=20k (bin005_full 0.4023): K no importa para el binario.
Queda a 0.0111 de la referencia.

### 4. WPC Idea 3 — barrido τ×K (momento 0.9, 6 épocas, eval bin)

Cinco celdas nuevas con el script 14 (la celda τ=0.05/K=20k es bin005_full,
0.4023, ya medida en la fase Morgan; la celda τ=0.05/K=50k es la prueba 3
de esta ronda, 0.4035):

| τ | K=20k | K=50k | picos que sobreviven al umbral |
|---|---|---|---|
| 0.05 | 0.4023 | **0.4035** | 18.9% / 18.4% |
| 0.10 | 0.3689 | 0.3696 | 7.5% / 7.2% |
| 0.20 | 0.3279 | 0.3281 | 2.5% / 2.4% |

**τ=0.05 es el mejor; subir τ empeora** porque el umbral descarta cada vez
más picos (a τ=0.20 sobrevive solo ~2.4% del nnz). K no mueve el número
(±0.001 entre 20k y 50k en las tres τ). **Ninguna celda supera 0.4146.**
MSE final: τ=0.10 → 0.01604/0.01607; τ=0.20 → 0.01703/0.01702.

### 5. WPC Idea 4 — re-ranking Tanimoto (script 22) — NO GANA

`22_rerank.py --tag lineal_bin` sobre el top-25 del lineal_bin (script 15
re-corrido: lineal_bin 0.414625, lineal crudo 0.372073 — reproduce la
referencia de la fase Morgan).

- **Paso A (verificación):** la huella Morgan recomputada con RDKit desde
  los SMILES del pool (mismo generador que 04: radio 2, 2048 bits) es
  **idéntica al pool en 137,882/137,882** moléculas distintas del top-25
  (0 sin SMILES, 0 fallos de parseo). El re-ranking por Tanimoto-bin es
  por tanto un no-op probado.
- **Control:** re-ordenar por Tanimoto con la predicción binarizada
  reproduce el orden original en **13,748/13,748** consultas y el mismo
  MRR 0.4146.
- **Paso B:** re-ordenar el top-25 por Tanimoto con la predicción cruda
  (clip [0,1]) **empeora: 0.4146 → 0.3874** (top-1 0.2598, top-5 0.5400;
  recall 0.8011 sin cambio — el conjunto top-25 es el mismo, solo cambia
  el orden).
- **Corrección de un bug propio (registrada):** la v1 de 22 calculaba el
  MRR promediando 1/rango solo sobre los hits, lo que inflaba los números
  ("antes" 0.5176, "crudo" 0.4836). Se detectó por el chequeo contra el
  0.4146 de 15 (0.4146/0.8011 = 0.5176 exacto). Corregido importando
  `mrr_from_ranks` de `casmi_lib` (media sobre TODAS las consultas);
  re-corrido en 56.1 s; el JSON lleva `mrr_corregido_2026_11_10: true`.
  Los números 0.5176/0.4836 sobreviven solo en las líneas del log del
  driver de la primera corrida — **no citarlos**.

### 6. Incidentes durante la ronda 2 (registro honesto)

1. **Mi `pkill -f` mató el entreno RGB a los ~2 min.** El patrón de
   `pkill -f` aparecía en la línea de comandos de mi propio shell y el
   ssh murió con exit 255; el driver sobrevivió pero el entreno RGB quedó
   en "Terminated" (log intento 1). El driver se relanzó limpio; el
   entreno es determinista (semilla 42), así que no se perdió ningún
   resultado, solo ~2 min. Lección: matar por PID, nunca con `pkill -f`
   con un patrón que esté en mi propia línea de comandos.
2. **Tags de evaluación crudo/bin.** El script 15 escribe el mismo
   nombre de JSON para la corrida cruda y la `--bin` con el mismo tag
   (la segunda sobrescribe a la primera). Se detectó antes de relanzar el
   driver y se corrigió usando tags `_bin` para las corridas `--bin`
   (misma convención que la ronda 1). El log del intento 1
   (`verificacion/ronda2_driver_intento1.log`) se preservó.

### 7. Archivos nuevos en `pruebas/` (ronda 2)

- Raíz: `20_rgb.py`, `21_entrenar_lineal_rgb.py`, `22_rerank.py`
  (corregido), `run_ronda2.sh` (driver), `wpc-schumann-ideas-cap37.md`.
- `intermedios/`: `X_train_rgb.npz` (271,047,441 B), `bins_rgb.npy`,
  `W_lineal_rgb20k.npy` (491,520,128 B), `P_val_rgb20k.npy` +
  `P_val_mol_rgb20k.npy`, `W_lineal_bin050k_mom.npy` (409,600,128 B) +
  P, `W_lineal_bin010_20k.npy` / `bin010_50k` / `bin020_20k` /
  `bin020_50k` + P (163,840,128 B los K=20k; 409,600,128 B los K=50k),
  `bins_piloto_<tag>.npy` correspondientes.
- `verificacion/`: `preparar_rgb.json`, `entrenar_lineal_rgb20k.json`,
  `entrenar_lineal_bin050k_mom.json`, `entrenar_lineal_bin010_20k.json`,
  `entrenar_lineal_bin010_50k.json`, `entrenar_lineal_bin020_20k.json`,
  `entrenar_lineal_bin020_50k.json`,
  `metricas_masa_rgb20k_val_mol.json`,
  `metricas_masa_rgb20k_bin_val_mol.json`,
  `metricas_masa_bin050k_mom_val_mol.json`,
  `metricas_masa_bin050k_mom_bin_val_mol.json`,
  `metricas_masa_bin010_20k_bin_val_mol.json`,
  `metricas_masa_bin010_50k_bin_val_mol.json`,
  `metricas_masa_bin020_20k_bin_val_mol.json`,
  `metricas_masa_bin020_50k_bin_val_mol.json`,
  `metricas_masa_lineal_bin_val_mol.json`,
  `metricas_masa_lineal_val_mol.json`,
  `metricas_rerank_lineal_bin_val_mol.json` (corregido),
  `ranks_masa_*_val_mol.npy` + `top_idx_masa_*_val_mol.npy`,
  `ranks_rerank_lineal_bin_val_mol.npy`, `ronda2_driver.log`,
  `ronda2_driver_intento1.log`, `22_rerank_corr.log`,
  `20_rgb_smoke.log`.

### 8. Pendiente / no hecho (actualizado tras la ronda 2)

- **RGB (0.4951) es el nuevo mejor de novo.** Falta decidir (Elán):
  notebook en runtime con W_rgb (491 MB) + huellas + índice de masa,
  y si se envía solo, en blend con baseline 1, o se deja el 0.146.
  Nada a Kaggle sin su decisión.
- **CE como feature:** falta el extracto por espectro desde train.parquet
  (esta caja); construirlo y subirlo solo con el OK de Elán.
- **Ideas WPC 3–4 cerradas por medición.** Queda la Idea 5 (higiene de
  resúmenes, no score) y los candidatos de baja prioridad (barrido fino
  de ancho de bin; deconvolución con la PSF) — no corridos en esta ronda.
- **Blend Morgan + baseline 1** sigue pendiente de decisión (ahora con
  RGB 0.4951 como alternativa/complemento posible).
- `datos/` sigue sin subirse; avisar antes si una prueba lo exige.
