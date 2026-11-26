# CHANGELOG — Enveda CASMI 2026

Historial verificado del proyecto. Fechas en hora central (CST).

## 2026-11-08

- **Baseline 1 completado ~23:35.** Búsqueda en librería espectral
  (bines 0.01 Da, top-100 picos, precursor ±10 ppm, coseno). Validación
  (semilla 42): spec-split MRR@25 = **0.8691** (top-1 0.8007, recall@25
  0.9865); mol-split MRR@25 = **0.0000 por construcción** (la molécula
  consultada no está en la librería; solo resuelve la clase de novedad
  1). Informe `kaggle-casmi-informe.md`.
- **submission.csv** (400 filas, md5 `7a7da389417fb162039a94394d76e1ee`).

## 2026-11-09

- **Paso 2 completado ~10:58.** Espectro → huella de Morgan (2048
  bits), base topológica del Genomic_agent. Veredicto: **kNN gana**
  (Tanimoto 0.3148, MRR@25 0.0346 mol-split); topológico v2 pierde
  (Tanimoto 0.1020, MRR 0.0148; peor que el nulo 0.2207). Lineal 0.2475
  Tanimoto pero peor MRR (0.0085). Informe
  `kaggle-casmi-informe-paso2.md`; métricas en
  `verificacion/metricas_paso2_*.json`.
- **submission_paso2.csv** (kNN, 400 filas, md5
  `ed758b99cdb3a91ac690f02fc3618779`). No enviado a Kaggle.
- **Repos creados ~09:45** (orden de Elán): este repo
  (`enveda-casmi-2026`, git local, sin remote). README con SHAs de los
  datos.
- **Respaldo en Drive ~11:49.** Respaldos Fractal Core / Enveda CASMI
  2026 / `respaldo-casmi-2026-11-09.zip` (12,294,600 B, verificado de
  vuelta; sin datos/ 3 GB ni intermedios/ 1.2 GB, re-descargables).
- **Envío 1 a Kaggle ~17:17** (notebook privado de copia estática,
  scriptVersionId 367850937): **"Submission Scoring Error"**, sin score.
- **Relleno a 25 ~19:20** (decisión de Elán "dále lo que pide"):
  0 de 33 filas tenían candidatos reales más allá del corte; las 33 se
  rellenaron repitiendo el top-1. `salida/submission_relleno25.csv`
  (md5 `7f082916a4a2bd68392750ea3cf2c8a7`). Informe
  `kaggle-casmi-informe-relleno25.md`.
- **Envío 2 a Kaggle 19:26:58** (scriptVersionId 367876111):
  **"Submission Scoring Error"** de nuevo.
- **Diagnóstico cerrado ~19:35.** La química está bien (0 fallos de
  10,000 bajo RDKit 2026.03.3, la versión pinned del scorer) y las
  filas cortas no eran la causa. Causa real: el test visible son
  ejemplos del train y Kaggle lo reemplaza por el test oculto en el
  re-run; el notebook estático entrega ids que no existen en el oculto.
  Fix propuesto: notebook que genere predicciones en runtime.
  `verificacion/diagnostico_scorer.md`.
- **Decisión de Elán ~19:38–19:39:** adelante con el fix en runtime, si
  no afecta al concurso; si pasa la verificación, que se suba.
- **Notebook en runtime construido y verificado ~19:48.**
  `casmi_runtime_notebook.ipynb` (7 celdas, 5 de código, autocontenido;
  build 297.5 s, total local 313.1 s, RSS 2,834 MB). Salida local byte
  por byte idéntica a submission_relleno25.csv (md5 7f082916…).
  Informe `kaggle-casmi-informe-runtime.md`; `verificacion/runtime.json`.
- **Envío 3 a Kaggle 20:04:39** (notebook casmi-runtime-notebook,
  Version 3, scriptVersionId 367881767, internet desactivado):
  **Succeeded, public MRR@25 = 0.146**. Corrida en Kaggle 431.8 s;
  md5 de salida 7f3fafa4… (≠ local, confirma test oculto). 3 de 5
  envíos diarios usados. 0/2 finales.
- **Fechas verificadas en Kaggle ~20:54** (orden de Elán): inicio
  14 sep 2026; entrada y fusión de equipos 7 dic 2026 11:59 PM UTC;
  envío final 14 dic 2026 11:59 PM UTC. Ambas versiones eran correctas:
  dos fechas distintas. `verificacion/fechas_casmi_verificadas.md`.
- **Evidencia del envío 3 archivada ~20:55** (auditoría H-4 del
  Coordinador): `verificacion/envio3_evidencia.md` (re-verificación con
  la sesión de Elán, sin cambios). Commit `1586902`.
- **Regla permanente de Elán ~21:00** (tablero §H): cada proyecto debe
  tener README + AGENTS.md y/o CLAUDE.md, todos CHANGELOG, y los
  serios logs de salida. Este AGENTS.md y CHANGELOG.md se crean por
  esa regla.

## 2026-11-10

- **GO de Elán ~11:33 ("Arranca"):** modelo espectro → huella Morgan para
  subir el 0.146 público. Trabajo local, nada a Kaggle/GitHub.
- **Entorno corregido ~11:40 (brief del coordinador):** el entorno oficial
  es el Python 3.12.3 del sistema (numpy 1.26.4), no el venv py313. Tras
  un reset de la caja se reinstalaron pyarrow==17.0.0, scikit-learn 1.6.1
  y rdkit==2024.03.6 (`verificacion/pip_instalacion_sistema.log`).
  **Validación RDKit:** 2024.03.6 vs 2026.9.1 idénticas en 500 SMILES
  (0 bits distintos de 1,024,000); el pool coincide bit a bit con el
  recomputo 2024.03.6. `verificacion/validacion_rdkit.json`.
- **Rutas reconciliadas:** los scripts 04–08 apuntan a
  `snt-taxonomia/kaggle-casmi`; los nuevos (12–17) usan
  `enveda-casmi-2026`. Los viejos no se ejecutaron contra el path viejo.
- **Filtro de masa de precursor (12–13) ~11:40:** candidatos = moléculas
  con algún espectro en full_lib dentro de ±10 ppm (piso 0.002 Da) de
  cualquier precursor de la consulta. Val_mol: mediana 106 candidatos,
  verdad sobrevive 1.0000. **MRR@25 mol-split con filtro: lineal_bin
  0.4146 (mejor), topo_v2 0.4123, lineal 0.3719 (era 0.0085, 43.7×), knn
  0.1406 (era 0.0346), oráculo 0.9937.** Spec-split: knn 0.8411,
  lineal_bin 0.4250, ens 0.8186. `verificacion/metricas_masa_*.json`.
- **WPC Idea 1 — binarizar el espectro (14–15) ~11:50–12:15:** el espectro
  como foto WPC (pico presente/ausente sobre τ). Piloto K=20k/80k/3ép:
  bin005 (τ=0.05) 0.3251 vs control 0.2886 (+12.6%). Presupuesto completo
  K=20k/261k/6ép: bin005_full_bin 0.4023 vs control 0.3985 (+0.95%).
  **No supera al lineal_bin K=50k (0.4146).** Mejora real pero pequeña.
- **Hallazgo de normalización (16):** la "L2" de 05/06 usa
  sqrt(Σ sqrt(int)), no la L2 verdadera (norma media de X_train 0.595).
  No afecta los MRR Morgan (misma transformación en entrenar y predecir);
  el 16 renormaliza a L2 verdadera para medir el coseno del blend.
- **Blend medido (16):** top-1 coseno verdadero ≥ 0.95 en mol-split
  8.82% (falso enrutamiento) y en spec-split 19.30%. Estimación blend
  con regla ≥ 0.95: ≈ 0.462 si el oculto es como el visible, ≈ 0.400 si
  es como el spec-split val; Morgan solo ≈ 0.416; ideal 0.492. **El blend
  es decisión de Elán (vía paralela, no decidida aquí).**
- **submission_morgan.csv (17) ~12:20:** lineal K=50k + lineal_bin +
  filtro de masa. 400 filas × 25 SMILES, md5
  `854584724275387743e744fd77757961`. 33 filas con relleno top-1.
  MRR esperado en oculto ≈ 0.416. Los tres submissions existentes
  intactos (md5 verificados). Informe `kaggle-casmi-informe-morgan.md`.
- **Logs movidos a `verificacion/`** (AGENTS.md regla 5; `logs/` eliminado).
  Nada a Kaggle ni a GitHub en esta fase.

## 2026-11-11

- **Envío 4 a Kaggle (RGB, ref 58256941): público MRR@25 = 0.049.**
- **Diagnóstico RGB cerrado.** H1 refutada; H2 confirmada (local 0.4951
  inflado ~6× por truth-in-pool); H3 confirmada. **Techo de retrieval en
  oculto ≈ 0.08–0.10 — no supera el 0.146; las clases 2/3 exigen
  GENERAR estructuras.** `verificacion/DIAGNOSTICO_RGB_0.049_REPORTE.md`
  (md5 `e4aeb736…`).
- **Recreación RGB bit-idéntica en Kali:** W md5
  `7a1c0347eb1d7e3a9734791c065c996b`, MRR bin 0.4951 exacto; config
  K=20k/canal, 6 épocas, batch 1024, lr 0.5, lam 0.0001, mom 0.9,
  semilla 42.
- **Repo de auditoría RGB subido y verificado:**
  https://github.com/Inzainos/casmi-rgb-audit (público, commit
  `ef4aa08b`, 32/32 archivos por blob SHA).

## 2026-11-12

- **Finales marcadas 2/2 DEFINITIVO ~18:32: envíos 3 (0.146) + 4
  (0.049).** PDF `verificacion/confirmacion_finales_2de2_2026-11-12.pdf`
  (204,552 B, md5 `3afec37f…`). Regla de Elán ~18:30: si un generativo
  supera los umbrales y mejora el público, reemplaza al envío 4; si no,
  las finales quedan.
- **Protocolo generativo de novo:**
  `verificacion/PROTOCOLO_GENERATIVO_DE_NOVO.md` (632 l.). Splits
  semilla 42: GEN-TRAIN 252,077; GEN-VAL-NOVEL 13,790; GEN-VAL-SPEC
  9,943. Umbrales aprobados por Elán: MRR_novel > 0.01 y Tanimoto@1
  > 0.267.
- **Fase 0 completa ~20:25:** tautomerización del pool en Kali (RDKit
  2026.03.3); auditoría de fuga: train∩val 152, train∩spec 81
  (tautómeros, no error de splits); **B1 MRR@25 MEDIDO 0.0019** (IC95
  0.0013–0.0025). Decisión de Elán ~20:33 (opción A): se mantiene
  GEN-VAL, 0.0019 como cero efectivo. Evaluador
  `30_evaluar_generativo.py` (autotest 6/6).
- **Piloto B (fingerprint → SMILES, Transformer 8,480,308 params):
  entrenamiento COMPLETO 20/20 en Colab GPU (Elán);** loss 2.3214 →
  0.9869; holdout greedy exact-match ep20 0.232 / val 0.833. GEN-VAL
  (n=2,000, beam 25): **MRR@25 = 0.3372** (IC95 0.31903–0.35512),
  Tanimoto@1 0.6373, validez 0.7467, unicidad 0.9679; novedad 0.9651
  (recompute local bit-idéntico). Ambos umbrales pasan por amplio
  margen. Paquete en Drive `casmi-piloto-b`.

## 2026-11-13

- **Bug del beam explicado:** en `evaluar_holdout`, `generar_beam`
  devuelve listas y el código compara lista vs string (l.506) →
  beam exact-match 0.0000 era un ARTEFACTO. **Fix aplicado local
  ~19:11:** `verificacion/piloto_b/entrenar_decodificador.py` (12 l.
  tras l.505, normalizar antes de comparar; md5 `18ec1200…`, 771 l.,
  34,469 B); test `test_fix_beam.py` + log. Sin commit en git (sin ops
  de git sin Elán).
- **Paquete v2-fix-beam + zip v2 integrado en Drive ~19:42:**
  `datos_piloto_b_colab.zip` (id `1RJWpmkRxxal0RTz4abmqCdfb5w-6vw-w`)
  ahora es v2 (30,298,704 B, md5 `ca8ac5a8…`); v1 preservada aparte
  (id `1TE_Lw…`, md5 `e4cef75a…`). Script con fix también en Drive
  (id `1Gq6Y8BK…`).
- **Prueba en Colab del fix PASADA (Elán, ~20:45; validada de Drive
  20:55–20:57):** beam exact-match 0.2525, greedy 0.2335, GEN-VAL
  MRR@25 0.3372 reproducido (IC95 0.31865–0.35515); `modelo.pt` en
  Drive ahora es el checkpoint reentrenado (md5 `2db2e48a…`,
  33,946,777 B). Veredicto: el fix funciona; 0.3372 confirmado.
- **Orden de Elán ~20:58: generar el notebook de envío con el fix,
  empaquetarlo y subirlo a Kaggle.** Notebook
  `verificacion/casmi_notebook_generativo_v1.ipynb` (35,703 B, 10
  celdas: híbrido baseline-1 + etapa 1 kNN + etapa 2 decoder beam-25)
  + constructor + manifiesto del dataset Kaggle (5 archivos,
  95,557,499 B ≈ 91.1 MiB, slug sugerido
  `casmi-generativo-pilotob-v1`). Smoke test decoder en CPU OK
  (beam-25, 2 moléculas, 191.8 s).
- **Constructor 1 falló ~21:23** (exec en segundo plano desapareció,
  sin entregar); continuación despachada ~21:24, **EN CURSO**:
  validación local en subconjunto ≤10 moléculas (run F en celda 6,
  decodificador; librería 2,438,064 filas; retrieval top1 1.0000;
  etapa 1 Tanimoto 0.6553). Nada a Kaggle hasta verificar (orden de
  Elán ~21:28: antes de subir, verificar que esté bien).
- **Temp de Drive eliminado ~22:27** (orden de Elán): 
  `W_lineal_rgb20k-transfer-TEMP.npy` (491,520,128 B, id
  `1KfpembTt7w8D40RFPTQf1H0tgWn1UQd3`) a papelera, verificado.

- **Envío generativo a Kaggle ~23:51 (orden de Elán 20:58 +
  verifica-primero 21:28; por navegador):** dataset privado
  https://www.kaggle.com/datasets/elnzainoscorona/casmi-generativo-pilotob-v1
  (5 archivos, 95.56 MB) + notebook privado
  https://www.kaggle.com/code/elnzainoscorona/casmi-notebook-generativo-v1
  (título distinto al dataset porque Kaggle rechazó el slug repetido).
  GPU T4 x2, internet OFF. Corrida exitosa en 450.4 s (7m 30s);
  librería 2,438,064 filas, 125,646,101 nnz (igual que local); salida
  400 filas, 25 SMILES/fila, 9/9 checks OK, md5 ac9da78d….
  **Público MRR@25 = 0.004 — NO supera el 0.146 ni el 0.049; las
  finales se quedan 3+4 (regla de Elán 2026-11-12).** Causa principal
  sospechosa: la rueda RDKit cp312 NO se instaló en el contenedor de
  Kaggle (Python 3.13) y el notebook corrió sin RDKit (dedup por
  SMILES exacto, sin filtro de validez). En el test visible el
  generativo no se disparó (0/400 con top1_cos < 0.95, media 1.0000,
  esperado).

- **Hallazgos de Datos sobre el 0.004 (2026-11-14 ~00:02, vía
  Coordinador; verificados en disco):** informe
  `verificacion/HALLAZGO_CASMI_GENERATIVO_0.004_RDKIT_2026-11-14.md`
  (249 l., md5 8393a2e3…), log `verificacion/kaggle_log_generativo_v1.txt`
  (162 entradas). La sospecha del RDKit es CORRECTA: la rueda cp312 no
  instaló en Kaggle ("RDKit NO disponible", "RDKit: False"); la versión
  exacta del contenedor NO está en el log ("Python 3.13" no verificado).
  Puntajes por conector: generativo ref 58333078 → 0.004; RGB ref
  58256941 → 0.049; envío 3 ref 58182345 → 0.146. Mecanismo: en el test
  visible el generativo no se disparó y la salida ES el retrieval del
  envío 3 (top-1 igual en 400/400); el 0.004 viene del re-run oculto,
  donde el generativo se dispara para la mayoría y sin RDKit los SMILES
  inválidos pasan sin filtro y el beam llena los 25 huecos sin respaldo
  de retrieval. Hueco real: run G nunca corrió la configuración de
  Kaggle (umbral forzado 1.01, RDKit: True). Diferencia clave: envíos 3
  y 4 no dependen de RDKit en runtime; el generativo es el único que
  sí. Opciones (no acciones): rueda compatible, loguear stderr de pip +
  sys.version, validar el fallback en local, blindar la fusión
  (generativo solo si HAS_RDKIT). Finales 3+4 intactas. Auditor
  pendiente.

- **Fix de librerías en curso (orden de Elán 2026-11-14 ~00:04: "inicia
  con ese fix que es más de librerías"):** rueda RDKit compatible.
  Subagente despachado: determinar el Python del contenedor de Kaggle,
  descargar la(s) rueda(s) RDKit 2026.03.3 que calce(n), actualizar el
  notebook (log sys.version + stderr de pip, probar ruedas en orden),
  validar en local y dejar dataset v2 + notebook v2 listos. NO subir a
  Kaggle sin GO de Elán. Investigación (Datos + Auditor) en paralelo.

- **Auditoría del Auditor Interno (2026-11-14 ~00:05, vía Coordinador;
  informe `auditoria/AUDITORIA_CASMI_GENERATIVO_0.004_2026-11-14.md`,
  167 l. / 24,421 B / md5 d295ef22…):** Datos CORRECTO, verificado línea
  por línea. Mecanismo de Datos correcto pero INCOMPLETO: no explica
  solo el 0.004 — el retrieval es idéntico al del envío 3 y debería
  haber conservado ~0.146 en clase-1; el 0.004 exige que el generativo
  también desplazara al retrieval en la mayoría de la clase-1 (portón
  top1_cos ≥ 0.95 se disparó donde no debía; medición propuesta §4 A-3).
  **Hallazgo principal: el GEN-VAL 0.3372 NO mide lo que se envió** —
  se midió con la huella EXACTA; el notebook usa la huella PREDICHA por
  la etapa 1 (Tanimoto ~0.31 en novelas), nunca medida; Fase 3 del
  protocolo pendiente; puerta G4 (híbrido honesto > 0.146) nunca se pasó
  antes del envío. El argumento del md5 del reporte del navegador es
  INVÁLIDO (run G fue subconjunto de 10). **Causa COMPUESTA:** (a) sin
  RDKit el generativo puntúa ~0 en nuevas + (b) el portón desplazó al
  retrieval en clase-1 oculta. Suscribe las 5 opciones de Datos y añade
  2: medir el híbrido honesto (G4) antes de cualquier re-envío y
  re-calibrar el portón 0.95. Finales 3+4. Pendiente: verificación
  cruzada por navegador de la página de submissions.

- **Addendum §7 del Auditor (2026-11-14 ~00:07, vía Coordinador;
  informe actualizado 179 l. / md5 d0a385f4…, verificación cruzada por
  navegador COMPLETA, solo lectura):** todo coincide en la página de
  Kaggle — líneas literales del RDKit iguales al log archivado (solo
  difieren los prefijos [tiempo|RSS]); sin línea "validez beam" (el beam
  nunca corrió, 0/400); generativo v1 Succeeded 0.004, enviado 13 nov
  23:31:24 GMT-0600, NO seleccionado, scriptVersionId 369395620;
  **finales 2/2 confirmadas: v3 0.146 (367881767, seleccionado), RGB v6
  0.049 (368622826, seleccionado).** Dato nuevo: **log del re-run
  oculto NINGUNO VISIBLE** — exactamente UNA corrida (la visible); la
  deducción del portón (A-3) queda como deducción. Pregunta abierta sin
  resolver: la procedencia exacta del artefacto puntuado no es
  observable desde la página del notebook (los puntajes se mostraron al
  enviar y en el visible el retrieval daría ~1.0, así que reflejan la
  verdad oculta); no afecta los veredictos.

- **Rediseño del envío generativo v2 (orden de Elán 2026-11-14 ~00:08:
  "haz el rediseño y antes de aplicarla se la pasas al analista y al
  auditor y hasta que te dé su go inicias").** Subagente despachado para
  escribir el diseño (`verificacion/REDISENO_GENERATIVO_V2_2026-11-14.md`,
  solo diseño, sin tocar código): 5 componentes — rueda compatible,
  blindaje de la fusión (generativo solo si HAS_RDKIT), re-calibración
  del portón 0.95, medición del híbrido honesto G4, validación local del
  fallback sin RDKit; con puertas de validación antes de cualquier
  re-envío. Flujo: diseño → revisión Datos + Auditor → solo con el GO
  de los dos se aplica. Nada a Kaggle sin decisión de Elán.

- **Investigación CERRADA (2026-11-14 ~00:10):** informe consolidado
  `verificacion/INVESTIGACION_BAJA_GENERATIVO_0.004.md` (300 l. /
  42,653 B / md5 121b59e7…, verificado por Mole en disco). Veredicto:
  causa compuesta — sin RDKit el generativo puntúa ~0 en nuevas +
  el portón 0.95 desplazó al retrieval en la clase-1 oculta + el
  GEN-VAL 0.3372 se midió con la huella EXACTA, no con la predicha.
  **Dato nuevo: el contenedor de Kaggle ES Python 3.13** (ruta
  `/usr/local/lib/python3.13/dist-packages/` en el stderr de nbconvert,
  log archivado líneas 159/161) → **la rueda compatible es la cp313**
  (pasado al fix de librerías). Reproducciones: Datos (top-1 400/400,
  conjuntos 399/400, orden 351/400, únicos/fila 24.1025); Auditor (diff
  por AST: score_query y binear_fila idénticas); 102 "SMILES Parse
  Error" en run G vs 0 en el log de Kaggle; cobertura local del portón
  (mol-split 8.82% ≥ 0.95, spec-split 19.30%); 5 md5s locales intactos.
  Pregunta abierta: procedencia exacta del artefacto puntuado no
  observable. Rediseño v2 + fix de librerías en curso.

- **Rediseño v2: diseño LISTO y pasado a revisión (2026-11-14 ~00:17):**
  `verificacion/REDISENO_GENERATIVO_V2_2026-11-14.md` (748 l. / 46,102 B /
  md5 2fa66bb9…, verificado por Mole en disco). 5 componentes + puertas
  G-V1–G-V7. Pasado al Coordinador para revisión de Datos + Auditor
  (orden de Elán: solo con el GO de los dos se aplica). Nota: el
  constructor ya trae la instalación multi-rueda del fix paralelo
  (`_buscar_ruedas`, log de sys.version). Nada aplicado aún; nada a
  Kaggle sin decisión de Elán.

- **Re-revisión del Auditor del rediseño v2 (2026-11-14 ~00:22, vía
  Coordinador; informe `~/workspace/auditoria/RE-REVISION_REDISENO_GENERATIVO_V2_2026-11-14.md`,
  191 l. / 20,227 B / md5 d2e29835…): SÍ, CON CORRECCIONES R-1–R-5.**
  Diseño sólido, casi todo coincide; GO condicionado a aplicar R-1–R-5
  (luego el Auditor verifica las líneas parchadas y el GO queda firme).
  R-1 (MEDIA): §5.3 cita mal la fuente de la muestra GEN-VAL de 2,000
  (el JSON no trae molecule_id; la real es predicciones.csv, 2,001
  líneas, o re-derivar con --muestra 2000, semilla 42). R-2 (MEDIA): el
  check blindaje_coherente es insatisfacible en la corrida normal activa
  (n_gen_first siempre 0 en el visible) → redefinirlo contra el conteo
  esperado por el portón. R-3: sys.version ya existe en el constructor
  :189 (criterio 1b: verificar, no añadir). R-4: :318 es un comentario;
  la línea es :360 o :653. R-5: cobertura_val.json mtime 2026-11-10,
  sin script generador; corregir atribución. El Auditor corrige su
  propia auditoría: Python 3.13 SÍ verificado; A-3 ya es medición
  (spec-split 19.30% ≥ 0.95 → ~80.7% de la clase-1 oculta habría ido al
  generativo). Subagente despachado para aplicar R-1–R-5. Revisión de
  Datos independiente, pendiente. G-V1 bloqueada hasta el informe del
  fix de la rueda. Nada se aplica hasta GO de los dos; nada a Kaggle
  sin decisión de Elán.

- **Veredicto de Datos sobre el rediseño v2 (2026-11-14 ~00:25, vía
  Coordinador): ACUERDO CON CORRECCIONES D-1–D-9.** Diseño sólido, casi
  todo coincide con sus recomputes; GO condicionado a aplicarlas (luego
  verifica las líneas parchadas, sin segunda revisión completa). D-1 =
  R-1 confirmada; D-2 = R-2 confirmada (suscribe la redefinición del
  Auditor); D-3 = R-3; D-4 = R-4; D-5 = R-5 con precisión: el script
  generador de cobertura_val.json SÍ existe (`16_cobertura_val.py`,
  línea 146). D-6 (MEDIA, nueva): §5.2 etapa B no ejecutable como está
  escrita — --huella solo exacta/corrupta y predecir busca la huella
  verdadera por molecule_id; la interfaz que sí existe es importar el
  decodificador como módulo y llamar dec.generar_beam con el tensor
  predicho. D-7: citar "§4, punto 5/4" (no §4.5/§4.4). D-8: fijar si la
  etapa A corre sobre qmol/qspec completos (23,684) o la submuestra de
  2,400. D-9: ranks_qspec_ppm_bm1.csv no existe; citar los existentes o
  el patrón ranks_{suf}.csv. Observación: retención ≥ 0.80 sobre 200
  moléculas (EE ~2.8%) — considerar el límite inferior del IC. D-1–D-9
  pasadas al subagente de correcciones (junto con R-1–R-5). G-V1
  bloqueada hasta el informe del fix de la rueda. Nada se aplica hasta
  GO de los dos; nada a Kaggle sin decisión de Elán.

- **Correcciones R-1–R-5 + D-1–D-9 APLICADAS al diseño (2026-11-14
  ~00:28):** diseño ahora 851 l. / 52,963 B / md5 c7e1b9e8…
  (verificado por Mole en disco + spot-checks). R-1/D-1 muestra GEN-VAL
  en predicciones.csv; R-2/D-2 blindaje = (n_gen_first == n_esperado);
  R-3/D-3 sys.version ya existe :189; R-4/D-4 chequeo final :360 y
  :653; R-5/D-5 atribución a 16_cobertura_val.py :146 (versión de
  Datos — la premisa de R-5 era incorrecta); D-6 etapa B fijada al
  módulo dec.generar_beam (CLI --modo predecir NO sirve); D-7 citas
  "§4, punto 4/5"; D-8 etapa A solo submuestra 2,400; D-9 patrón
  ranks_{suf}.csv; (3c) retención sobre el límite inferior de Wilson
  95%. S-1/S-2/S-3 no tocadas. Pasado al Coordinador: el Auditor
  verifica R-1–R-5 y Datos D-1–D-9; cada GO queda firme cuando
  verifiquen. G-V1 bloqueada hasta el informe del fix de la rueda.
  Nada se aplica hasta GO firme de los dos; nada a Kaggle sin decisión
  de Elán.

- **GO FIRME DEL AUDITOR sobre el rediseño v2 (2026-11-14 ~00:31, vía
  Coordinador; addendum §8 en su informe, ahora 208 l. / 23,743 B /
  md5 298b67f3…, verificado por Mole en disco: md5 y líneas exactos,
  sección §8 presente).** Verificó las líneas parchadas de R-1–R-5 en
  disco (diseño 851 l., md5 c7e1b9e8…): R-1 en §5.3 (:578–588,
  predicciones.csv + re-derivación; 0 restos de la cita vieja); R-2 en
  §3.4 (:314–328, blindaje = (n_gen_first == n_esperado), propagada a
  (2a)/(2b) y G-V6; la vieja solo como nota histórica); R-3 en :265 y
  :270 (sys.version ya existe :189); R-4 en :295–296 y :266 (:360 y
  :653); R-5 en :122 (16_cobertura_val.py :146 — verificado por el
  Auditor: script existe, 7,495 B, línea 146 escribe el JSON; la
  versión de Datos es correcta). Números clave intactos; el total v2
  132,744,644 B vive en el manifiesto (:63, intacto), no en el diseño.
  S-1/S-2/S-3 no tocadas. **El diseño queda aprobado por el Auditor.**
  Datos verifica las D-1–D-9 por su lado; nada se aplica hasta el GO
  firme de los dos; nada a Kaggle sin decisión de Elán. G-V1 sigue
  bloqueada hasta el informe del fix de la rueda.

- **GO FIRME de Datos al rediseño v2 (2026-11-14 ~00:37, vía
  Coordinador):** verificó sus D-1–D-9 línea por línea en el diseño
  parchado (851 l. / md5 c7e1b9e8…): D-1 (:578–588), D-2 (:314–328,
  :347–352, :686), D-3 (:265, :270), D-4 (:266, :295–296), D-5 (:122),
  D-6 (:520–535), D-7 (:399, :436, :587), D-8 (:510–513), D-9
  (:368–369). Textos viejos: 0 restos. Observación (3c) aplicada
  (:439–450). Números clave intactos. El Auditor ya dio su GO FIRME.
  **Ambos revisores con GO FIRME — por orden de Elán se INICIA LA
  APLICACIÓN.** Subagente despachado: blindaje, re-calibración del
  portón, G4 (etapa A local + paquete de etapa B para Elán en Colab),
  validación del fallback, notebook v2 completo. El re-envío a Kaggle
  sigue siendo decisión aparte de Elán; nada se sube sin su GO. G-V1
  bloqueada hasta el informe del fix de la rueda.

- **Fix de librerías: rueda RDKit cp313 (2026-11-14, orden de Elán
  "inicia con ese fix que es más de librerías"):** el contenedor de
  Kaggle es Python 3.13 (verificado en el log de la corrida v1, rutas
  `/usr/local/lib/python3.13/`, líneas 159/161) — la rueda cp312 no podía
  instalar. Añadida la rueda `rdkit-2026.3.3-cp313` (37,187,145 B, md5
  `a9699b01…`, sha256 verificado contra PyPI) junto a la cp312.
  Notebook v2 (`casmi_notebook_generativo_v2.ipynb`, md5 `0b5b3a46…`):
  imprime `sys.version`, prueba las ruedas en orden (la de este Python
  primero) y loguea returncode + stderr de pip por intento; celdas 2–8
  idénticas a la v1. Validación local: prueba aislada PASA (cp312 falla
  con "not a supported wheel", cp313 instala, RDKit 2026.03.3 funcional)
  + run V2b TODO OK 9/9 en 1,170.1 s con RDKit: True, salida
  BYTE-IDÉNTICA a la del run G (md5 `5be7ad54…`). Dataset v2: 6 archivos,
  132,744,644 B. Reporte `verificacion/FIX_RUEDA_RDKIT_2026-11-14.md`.
  **La rueda es NECESARIA pero NO SUFICIENTE (Auditor): antes del
  re-envío faltan G4 (híbrido honesto) + re-calibrar el portón 0.95
  (decisión de Elán). Nada subido a Kaggle.**

## 2026-11-19

- **Checkpoint por época + resume en `verificacion/entrenar_decodificador_v3.py` (orden de Elán 15:38:51, vía Coordinador).** Al final de cada época se guarda `checkpoint_epoca_NN.pt` (modelo + optimizador AdamW + época + paso global + historial + estados RNG numpy, torch CPU y torch CUDA) en el directorio de salida; `--resumir` reanuda desde el último checkpoint con la misma trayectoria (el plan de LR depende del paso global y la corrupción/permutación de los RNG, todos restaurados); con argumentos incompatibles con el checkpoint, aborta en voz alta. El checkpoint final `modelo.pt` se guarda como antes, antes de la evaluación final. Script md5 `da9ab88608e5d5124ec75f8e366bdd2d` (antes `be8b5175…`), idéntico local y Kali. Retención: un checkpoint por época (cada uno ~97 MB: modelo 34 MB + optimizador ~64 MB; 20 épocas ≈ 2 GB; disco Kali 687 GB libres). Sin cambios a la matemática, la corrupción (perfil bin05, keep frecuencia, p_corrupt 0.5) ni la evaluación. **Verificación (corrida seca, Kali CPU, n_train 500, 3 épocas, batch 64):** control ininterrumpido + corrida matada tras la época 2 y reanudada con `--resumir` — `metricas.json` IDÉNTICAS (historial 2.7976 → 2.2604 → 2.1273, holdout final igual) y `modelo.pt` byte-idéntico (md5 `306a21ba…`). **La corrida completa actual (PID 5599, época 17/20) sigue con el script previo sin checkpoints; el checkpoint por época aplica a los re-lanzamientos.** Incidencia de la prueba: matar con `kill` al PID del wrapper `nohup` no mata al proceso python hijo (sobrevivió y terminó solo; dos escritores en el mismo directorio, contenido idéntico por determinismo) — lección: matar por PID exacto del proceso python, no del wrapper. Artefactos de prueba en la Kali: `/home/deamon/casmi-sonda-v3/prueba_ckpt_control/`, `prueba_ckpt/`, `prueba_ckpt2/` (+ logs).
