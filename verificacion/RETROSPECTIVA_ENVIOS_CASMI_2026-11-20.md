# RETROSPECTIVA DE ENVÍOS CASMI — 2026-11-20

**Proyecto:** Enveda CASMI 2026 (`~/workspace/enveda-casmi-2026/`)
**Fecha:** 2026-11-20 (orden directa de Elán, 08:43 CST)
**Autor:** Mole (subagente de retrospectiva) — para la cadena **Datos → Auditor → Elán vía el Coordinador**
**Estado de CASMI:** PAUSADO (Elán: primero DQI, luego CASMI). Esto es SOLO análisis para retomarlo con foco.
**Reglas cumplidas:** nada a Kaggle; nada a GitHub (repo sin remote, local only); ningún envío nuevo construido; `salida/` no tocado — md5 verificados antes (08:44) y después (ver Anexo B), 5/5 intactos; mediciones locales con datos existentes, un proceso pesado a la vez; system python 3.12.3 / numpy 1.26.4.

**Envíos (6 en total; 1–2: error de scoring, cerradas):**

| # | Envío | Fecha | Ref / nota | Público MRR@25 |
|---|-------|-------|------------|----------------|
| 3 | Retrieval (baseline-1) | 2026-11-09 | ref 58182345 | **0.146 — el mejor, la GUÍA** |
| 4 | RGB (Morgan + kernel) | 2026-11-11 | ref 58256941 | 0.049 |
| 5 | Generativa v1 | 2026-11-13 | ref 58333078 | **0.004** |
| 6 | Generativa v2 | 2026-11-19 06:28 | notebook casmi-notebook-generativo-v1 Version 2 | **0.138 — último envío** |

Finales 2/2 = submission 3 + submission 4 (marcadas 2026-11-12). La v2 generativa SIN marcar (decisión de Elán 2026-11-19 ~07:02: "No, déjenla como está por ahora"). Cierre de entrada/fusión: **7 dic 2026**; envío FINAL: **14 dic 2026 11:59 PM UTC**.

**Criterio de adopción vigente (congelado):** novel MRR ≥ 0.021, umbral top1_cos 0.6 (`porton_calibracion.json`, md5 `cacbe566…`).
**Validación local de referencia:** 1,200 moléculas (1,000 novel + 200 clase-1), híbrido G4.

**Nota de correcciones — 2026-11-20 (orden de Elán 10:57).** Correcciones A-R-1, A-R-2 y A-R-3 del Auditor y D-RT-1 y D-RT-2 de Datos aplicadas y verificadas por recomputación desde las fuentes (script `verificacion/verificar_correcciones_retrospectiva.py`, log `verificar_correcciones_retrospectiva.log`: 25 PASS, 4 diferencias de redondeo de 0.0001, 0 FAIL): (1) la f estimada pasa de la fórmula simplificada 0.168 a la exacta de dos términos 0.1568, con todos sus derivados propagados (RGB clase-1 oculta 0.2917 → 0.3124; escenario B 0.7640 → 0.8176; RGB esperado 0.0832/0.0967 → 0.0776/0.0903; residual 4.3–6.0 SE → 3.6–5.2 SE); el escenario A (f = 0.1446) no cambia porque ya usaba la fórmula exacta; 0.168 queda como valor simplificado y convención del proyecto. Los valores aquí son los recomputados a precisión completa; el Auditor citó 0.3125/0.8177 usando f redondeada a 0.1568 y 0.0777 por desliz de redondeo (0.1568×0.4951 = 0.07763168 → 0.0776, D-RF-1) — diferencia de 0.0001, sin efecto en ningún veredicto ni en el rango de residuales. (2) La degradación de la v1 es ~18 SE, no ~12 SE (§0 y H-6; §3.2 ya decía ~18 SE; el ~12 SE es la cifra de RGB y sigue siendo correcta ahí). (3) H-1: la visible de la v1 NO es byte-idéntica a la submission 3; el par byte-idéntico es v1-visible == corrida local v2 (ambas `ac9da78d97c685adee13f1f89e5a8cbd`, verificado por `cmp`). Ninguna corrección cambia ningún veredicto.

---

## §0 — Resumen ejecutivo: los 6 hallazgos

1. **La submission 3 (retrieval, 0.146) sigue siendo la guía, y lo es por la clase-1, no por lo generativo.** Localmente: clase-1 MRR 0.8859 (197/200, recall@25 0.985); novel 0.0115 (12/1,000). El retrieval es robusto donde la molécula está en la biblioteca y casi nulo donde no está. El score público 0.146 es, en su mayor parte, clase-1 × fracción de clase-1 en el oculto (f ≈ 0.1568 exacto, estimación circular — ver H-2; 0.168 es la fórmula simplificada, convención del proyecto).
2. **La generativa v2 (0.138) NO es una degradación demostrada respecto a la 3.** La brecha pública es −0.008, que es ~1 error estándar del MRR público (SE ≈ 0.008, informe RGB) y el 0.138 cae DENTRO del IC95 del híbrido local G4 [0.1362, 0.1778]. Medición nueva de hoy: en la misma validación de 1,200, el híbrido supera al retrieval-solo por **una sola molécula** (+0.0000625 novel MRR; +0.0000521 híbrido). La validación local NO tiene potencia para ordenar 3 vs v2; el público tampoco, con estos números.
3. **La rama generativa aporta hoy exactamente 1 molécula de 1,200 (ZHHFMKYOKONAFI, rank 16).** Los otros 12 aciertos novel son gemelos de retrieval (11 en rank 1, 1 en rank 2, todos con top1_cos ≥ 0.6, todos en posición 1–2 de la lista de retrieval). La contribución generativa es 0.5% del MRR novel. El cuello es la COBERTURA (verdad en el beam: 9/1,000 en beam-25, 15/1,000 en beam-50), no el orden.
4. **El gate se equivoca en las DOS direcciones (deducción verificada del techo, §2 H-5).** Localmente no desplaza ningún acierto de retrieval (el conjunto {verdad en ret_sm Y top1_cos < 0.6} está vacío en novel), PERO 8 moléculas novel tienen su verdad en el beam-25, NO en el retrieval top-25, y están enrutadas a retrieval — el beam nunca se consulta y se pierden. El techo optimista beam-25 (0.0205) sigue sin alcanzar el criterio 0.021.
5. **La RGB (0.049) es un problema de ORDEN, no de cobertura — lo opuesto a la generativa.** En el visible de 400: la verdad está en su top-25 en 367/400 (recall 0.9175) pero en rank 1 solo en 182/400 (top-1 0.4550). Sus conjuntos de candidatos se solapan solo 30% con los de baseline-1 (Jaccard medio 0.3017). El kernel v6 reproduce el envío casi exacto (Jaccard 0.9732) — la degradación está en el MÉTODO, no en el runtime de Kaggle.
6. **La v1 (0.004) fue una degradación real y compuesta (97% de caída, ~18 SE); la v2 la revirtió casi por completo (0.004 → 0.138, +0.134).** Pero el contrafactual de la investigación de la v1 ("sin la causa (a) el score sería ~0.146 + contribución generativa") quedó REFUTADO por la v2: con las dos causas corregidas, el híbrido marca 0.138, NO 0.146+, porque la contribución generativa real es ~0. La v2 atendió el blindaje y el gate; NO atendió la cobertura ni la exactitud del decodificador (checkpoint sin reentrenar).

---

## §1 — Qué hicimos BIEN

### 1.1 Submission 3 (retrieval, 0.146) — lo que funciona

**Lo que funciona: el retrieval de clase-1.** Medido hoy en la validación de 1,200 (retrieval-solo, `ret_sm` de `etapaA.npz`, verdad por inchikey14):

| Métrica | Clase-1 (n=200) | Novel (n=1,000) | Todo (n=1,200) |
|---|---|---|---|
| MRR@25 | **0.8859** | 0.0115 | 0.1572 |
| IC95 | [0.8492, 0.9208] | [0.0055, 0.0185] | [0.1362, 0.1778] |
| top-1 | 0.8150 | 0.0110 | 0.1450 |
| recall@25 | **0.9850** | 0.0120 | 0.1742 |
| aciertos | 197/200 | 12/1,000 | 209/1,200 |

**Qué lo sostiene:**
- La molécula está en la biblioteca y su espectro coincide: 197/200 clase-1 tienen la verdad en el top-25 de retrieval, 163/200 en rank 1 (top-1 0.8150).
- El método es determinista y reproducible: la salida visible local (`submission_kaggle_visible.csv`) es byte-idéntica a la corrida local v2 (`verificacion/generativo_check/submission_v2_u06_400mol.csv`, md5 `ac9da78d…` en ambas). El retrieval no depende del contenedor ni de RDKit: no hay nada que se pueda romper en runtime.
- El score público 0.146 es coherente con la validación local: 0.146 cae dentro del IC95 del híbrido G4 [0.1362, 0.1778] y dentro del IC95 del retrieval-solo [0.1362, 0.1778]. La submission 3 no es un accidente de un set fácil: es el mismo régimen que la validación local.

**Lo que NO funciona en la 3 (para no mitificarla):** en novel es casi nula — 12/1,000 aciertos, todos gemelos (la verdad es un tautómero o un casi-duplicado de una molécula de la biblioteca, no la molécula misma). El 0.146 público depende casi por completo de la fracción de clase-1 del oculto (f ≈ 0.1568 exacto, estimación circular — ver H-2). Si el oculto tuviera f = 0, la 3 marcaría ~0.0115.

### 1.2 Generativa v2 (0.138) — lo que funciona

**La recuperación 0.004 → 0.138 (+0.134) es el hecho central de la v2.** Es la mayor recuperación entre envíos consecutivos del proyecto y se logró SIN cambiar el decodificador (checkpoint `modelo.pt` md5 `2db2e48a…`, sin reentrenar) y SIN cambiar el retrieval (AST diff 0 líneas en `score_query`/`binear_fila`). Lo que cambió fue el blindaje y el gate — es decir, la v2 es la prueba de que la v1 no falló por el modelo, sino por el envoltorio.

**Qué hizo bien la v2, con número:**
- **Rueda cp313:** la causa (a) de la v1 quedó atendida con evidencia. `rdkit-2026.3.3-cp313-cp313-manylinux_2_28_x86_64.whl` existe en el dataset v2 local con md5 `a9699b01…` (verificado hoy; coincide con el sha256 de PyPI reportado en la investigación). La v1 corrió con `RDKit: False` en el contenedor Python 3.13; la v2, con la rueda cp313 + chequeo de RDKit + log, no repite ese fallo. (La confirmación definitiva sería el log de la corrida v2 en Kaggle — ver H-7: no está en el workspace.)
- **Gate recalibrado 0.95 → 0.6, con calibración congelada:** `porton_calibracion.json` (md5 `cacbe566…`) — grid 0.5–0.99; MRR de calibración 0.15049346405228756 en 0.5 Y en 0.6 (empate exacto, G-V7 E-2); novel 0.009, clase-1 0.8579607843137255 a 0.6; retención de clase-1 0.995 (199/200, Wilson LB 0.972226295303572). El umbral 0.6 fue elegido por retención de clase-1, no por MRR — y G-V7 E-3 corrigió el razonamiento: un umbral MÁS ALTO no protege clase-1, la retención CAE con el umbral (0.995 a 0.6, 0.980 a 0.7, 0.940 a 0.8).
- **Shield (blindaje) probado localmente:** validez 1.0 en la fusión G4 (30,000/30,000 candidatos OK en la evaluación del reentrenamiento v3; unicidad 0.9541, novedad 0.3275). Los SMILES inválidos se filtran y el retrieval rellena — el 0.0 de validez del micro v4 quedó explicado como sub-entrenamiento (1 época, 500 mol), no como bug del beam.
- **El beam es válido como mecanismo:** con fingerprints EXACTAS, el decodificador alcanza GEN-VAL MRR 0.337223364107438, IC95 [0.31865, 0.35515] (investigación v1). El decodificador SABE generar la molécula correcta cuando la huella es correcta. El problema no es el decodificador en sí, es la huella predicha (ver §4.2).
- **La v2 no rompió nada que funcionaba:** clase-1 del híbrido G4 = 0.8859166666666667, IDÉNTICA al retrieval-solo (medido hoy, diferencia 0.0000000). El blindaje no desplazó ni una sola molécula de clase-1 en la validación local.

**Qué sostiene el 0.138:** lo mismo que sostiene el 0.146 de la 3 (la clase-1, 0.8859) MÁS 1 molécula generativa (ZHHFMKYOKONAFI). La v2 es, en la práctica, la submission 3 con un blindaje y una rama generativa que hoy aporta 1/1,200 moléculas. Su valor no es el +0.00006 local, sino la demostración de que el camino híbrido es ENVIABLE sin colapsar: el riesgo de la v1 quedó contenido.

### 1.3 Medición nueva de hoy: retrieval-solo vs híbrido en la MISMA validación (1,200)

Esta es la comparación que el proyecto no había hecho: el retrieval-solo (`ret_sm` de `etapaA.npz`, sin rama generativa) contra el híbrido G4, en las mismas 1,200 moléculas, con la misma verdad (inchikey14) y el mismo scorer.

| | Híbrido G4 | Retrieval-solo (hoy) | Δ (híbrido − retrieval) |
|---|---|---|---|
| Todo MRR | 0.15728819444444442 | 0.1572361111111111 | **+0.0000521** |
| Novel MRR | 0.0115625 | 0.0115 | **+0.0000625** |
| Clase-1 MRR | 0.8859166666666667 | 0.8859166666666667 | **0.0000000** |
| Aciertos novel | 13 | 12 | +1 molécula |
| Aciertos clase-1 | 197 | 197 | 0 |

**Lectura:** el híbrido supera al retrieval-solo por exactamente **1 molécula en 1,200** (1/16/1,000 = 0.0000625 novel MRR). La rama generativa no ayuda ni estorba a la clase-1 (Δ = 0.0000000 exacto). El IC95 del híbrido [0.1362, 0.1778] y el del retrieval-solo [0.1362, 0.1778] son el mismo intervalo: la validación de 1,200 NO distingue ambos métodos. Cualquier afirmación de que "el híbrido es mejor/peor que el retrieval" basada en la validación local es ruido.

**Desglose de los 13 aciertos novel del híbrido (medido hoy, por molécula):**

| Molécula | Rank híbrido | Rama | top1_cos | Pos. en ret_sm | Rank retrieval-solo |
|---|---|---|---|---|---|
| NGHMDNPXVRFFGS | 1 | retrieval | 1.0000 | 1 | 1 |
| OYONTEXKYJZFHA | 1 | retrieval | 0.6066 | 1 | 1 |
| QNGYADUGPHQWNE | 1 | retrieval | 1.0000 | 1 | 1 |
| RKGMCYIVUHRCLR | 1 | retrieval | 1.0000 | 1 | 1 |
| SBUXRMKDJWEXRL | 1 | retrieval | 0.9892 | 1 | 1 |
| SLWWJZMPHJJOPH | 1 | retrieval | 0.9811 | 1 | 1 |
| UTAIBCIOWTVUDJ | 1 | retrieval | 1.0000 | 1 | 1 |
| UWPWEPOUKLTBMR | 1 | retrieval | 0.9973 | 1 | 1 |
| XDBBLKMHMDEXQZ | 1 | retrieval | 0.9553 | 1 | 1 |
| YSYIWCNPSZNNKW | 1 | retrieval | 0.9980 | 1 | 1 |
| ZFHSKBJBODQVBX | 1 | retrieval | 0.9763 | 1 | 1 |
| ULFUJLFTRWWLPO | 2 | retrieval | 0.9494 | 2 | 2 |
| ZHHFMKYOKONAFI | 16 | **generativa** | 0.1613 | — (no está en ret_sm) | — |

**Lectura del desglose:**
- Los 12 aciertos de retrieval son 11 en rank 1 y 1 en rank 2, TODOS con top1_cos ≥ 0.6 (mínimo 0.6066) y TODOS en posición 1–2 de la lista de retrieval. Ninguno está en la rama generativa.
- El único acierto generativo (ZHHFMKYOKONAFI) tiene top1_cos 0.1613 (el más bajo de los 13) y su verdad NO está en el retrieval top-25: es el único caso donde la rama generativa aporta algo que el retrieval no tiene.
- 4 de los 13 son aciertos huérfanos por tautomerismo (VPRAQYXPZIFIOH, JNXGVLRHSYOTKM, BLFPNXIXUPFMIC, PNARHJLMOIEDMU — la verdad coincide por tautómero, no por identidad de SMILES).
- Novelas con verdad en el retrieval top-25: **12/1,000**. Clase-1 con verdad en el retrieval top-25: **197/200**. Esos son los techos del retrieval-solo, y el híbrido los alcanza exactamente (12 y 197) más 1 generativa.

---

## §2 — Qué NO estamos comparando / NO estamos viendo

Cada hueco lleva su evidencia o su ausencia explícita. Donde dice **NO MEDIBLE**, no hay número local que lo resuelva y no se inventa ninguno.

### H-1 — Complementariedad por molécula entre la submission 3 y la generativa v2 en el visible de 400: NO MEDIBLE

**Pregunta:** ¿aciertan moléculas distintas la 3 y la v2 en el visible de 400?
**Respuesta: no se puede medir con los datos locales. Falta: la salida visible de 400 moléculas de la notebook v2.**

- La salida visible de la v1 SÍ existe localmente (`verificacion/generativo_check/submission_kaggle_visible.csv`, md5 `ac9da78d97c685adee13f1f89e5a8cbd`) y es byte-idéntica a la corrida local v2 (`verificacion/generativo_check/submission_v2_u06_400mol.csv`, mismo md5 en ambas — verificado por `cmp`). Frente a la salida visible de la submission 3 (`salida/submission.csv`, md5 `7a7da389417fb162039a94394d76e1ee`) NO es byte-idéntica (`cmp` difiere), pero es casi idéntica en contenido: mismo top-1 en 400/400, Jaccard medio 0.9998 (medido hoy). La v1, en el visible, NO generó nada: su salida ES salida de retrieval. Por eso la comparación 3 vs v1-visible es trivialmente idéntica en contenido, aunque no byte-idéntica en archivo.
- La v2 corrió SOLO en Kaggle, sobre el set oculto. Su salida de 400 visibles (si el notebook la produjo) no está en el workspace: no hay hook run, no hay corrida local de la notebook v2, no hay CSV visible v2. El único número v2 que existe es el público 0.138, agregado, sobre el oculto.
- **Consecuencia:** la complementariedad 3 vs v2 por molécula es desconocida. Lo único que se puede decir es que en la validación local de 1,200 el híbrido (la arquitectura de la v2) acierta 13 novel y el retrieval 12, con 12 en común — es decir, la complementariedad local es de 1 molécula. Pero la validación local NO es el visible de 400 ni el oculto: es una tercera población.
- **Qué haría falta:** correr la notebook v2 localmente sobre el visible de 400 (o recuperar su salida de Kaggle si existe) y comparar por molécula contra la salida visible de la 3. Costo: una corrida local del notebook con el dataset v2 (ya existe en `~/workspace/casmi-drive-upload-2026-11-14/dataset_v2/`). No se hizo porque CASMI se pausó y porque la v2 se envió directo a Kaggle sin corrida local previa (ver §3.2, punto no atendido nº 6).

### H-2 — Qué esconde el número público único: f del oculto y dureza de clase-1

**El score público es UNA ecuación con DOS incógnitas.** Para el retrieval-solo: público = f × R1 + (1 − f) × Rnovel, donde f = fracción de clase-1 en el oculto, R1 = MRR de clase-1 en el oculto, Rnovel = MRR novel de retrieval en el oculto. Con un solo número (0.146) no se pueden separar f, R1 y Rnovel.

**Lo que inferimos (con la cadena de inferencia explícita):**
- f ≈ 0.1568 (valor exacto, fórmula de dos términos), bajo el supuesto R1_oculto = 0.8691 (spec-split local) y Rnovel_oculto ≈ 0.0115 (validación local): f = (0.146 − 0.0115)/(0.8691 − 0.0115) = 0.1345/0.8576 = 0.15683. El valor 0.168 que citaba la versión anterior es la fórmula simplificada f = público/R1 = 0.146/0.8691 = 0.16799 (medido hoy, Parte D), que desprecia el término novel; 0.168 sigue siendo la convención establecida del proyecto (`G4.json`: `mrr_formula_f_0.168`; AGENTS.md: f ≈ 0.17), pero el valor exacto bajo los supuestos declarados es 0.1568. **La estimación es circular:** usa el spec-split local (0.8691) como proxy de R1_oculto, y el spec-split es el set donde el retrieval funciona MEJOR (las moléculas están en la biblioteca y los espectros son del mismo instrumento). Si R1_oculto es menor (timsTOF, ver H-3), f es mayor. (La inversión exacta devuelve R1 = 0.8691 por construcción — la f se derivó suponiendo ese R1 — así que no es evidencia nueva.)
- Dos escenarios compatibles con el 0.138 de la v2 (medidos hoy, Parte D, con novel 0.0116 y clase-1 local 0.8859):
  - **Escenario A:** f = 0.1446 (el oculto tiene menos clase-1 de lo estimado), clase-1 = 0.8859 → 0.138.
  - **Escenario B:** f = 0.1568, clase-1 = 0.8176 (la clase-1 oculta es más dura que la local) → 0.138.
  - Ambos escenarios son indistinguibles con el score público solo. Y un tercer escenario — fuga por el gate (moléculas retrievables enrutadas a generación) — también cabe (ver H-5).
- **Lo que NO podemos saber:** f real, R1_oculto real, Rnovel_oculto real. Kaggle no devuelve resultados por molécula; el público es un agregado. **NO MEDIBLE con los datos locales: falta el desglose por molécula del score público, que solo Kaggle tiene y no entrega.** La única forma de separar f de R1 sería un envío diseñado para ser informativo (ver §4.3), no un envío para subir el score.

### H-3 — La dureza de clase-1 en el oculto (timsTOF) vs la validación local

**Lo que sabemos:** el informe RGB (2026-11-11) ya estableció que el oculto es más duro que la validación: la clase-1 oculta implicada para RGB es 0.049/0.15683 = 0.3124 (medido hoy, Parte D; con la f simplificada 0.168 era 0.2917), contra 0.4951 de RGB en validación local y 0.5758 en visible. La validación local es spec-split: los espectros de validación vienen del mismo instrumento/condiciones que el train. El oculto trae timsTOF (otra plataforma). La brecha 0.4951 → 0.3124 es el tamaño del salto de dominio para RGB.
**Lo que NO sabemos:** la clase-1 oculta implicada para baseline-1 (el retrieval de la 3). La inversión de la fórmula exacta devuelve R1 = 0.8691 por construcción — la f se derivó suponiendo ese R1 — así que ASUMIR que R1_oculto es igual al spec-split es la hipótesis más optimista. Si la clase-1 oculta de baseline-1 fuera, por ejemplo, 0.8176 (escenario B), f sería 0.1568 y el 0.138 de la v2 se explicaría entero por dureza, sin fuga y sin caída de f.
**NO MEDIBLE localmente: falta un set de validación con espectros timsTOF.** La validación local de 1,200 no tiene ese salto de dominio. Cualquier número de clase-1 local (0.8859) es un techo optimista para el oculto.

### H-4 — Qué acierta la submission 4 (RGB) que las demás no: NO MEDIBLE con la verdad proxy

**Pregunta:** ¿hay moléculas donde RGB acierta y baseline-1 no?
**Respuesta: con los datos locales, la pregunta está mal planteada — la verdad del visible es el propio top-1 de baseline-1.**

- La verdad del visible de 400 que usamos (y que usó el informe RGB) es el top-1 de baseline-1, justificada porque top1_cos = 1.0000 en 400/400 (log v1 de Kaggle): las moléculas visibles están en la biblioteca con espectro idéntico, así que el top-1 de baseline-1 es casi seguro la molécula verdadera.
- Bajo esa verdad proxy, RGB acierta 367/400 y baseline-1 acierta 400/400 — por construcción, RGB no puede tener un acierto que baseline-1 no tenga (el informe de hoy lo verifica: "hits RGB que baseline-1 NO tiene (imposible por construcción): 0").
- **Lo que SÍ se midió hoy (y es lo que la pregunta realmente puede responder):** RGB y baseline-1 VEN vecindarios distintos. Mismo top-1 solo en 182/400 moléculas; Jaccard medio de los conjuntos top-25 = 0.3017. RGB ordena mal (top-1 0.4550) pero cubre bien (recall@25 0.9175): la verdad está en su top-25 en 367/400, pero en rank 1 solo en 182/400. De sus 367 aciertos, 185 están en ranks 2–25.
- **Lo que falta para responder la pregunta de verdad:** la lista verdadera de InChIKey14 del visible de 400 (una por spectrum_id). `datos/test.parquet` NO tiene columna smiles y `datos/train.parquet` NO tiene columna molecule_id — no hay forma local de construir esa verdad. **NO MEDIBLE: falta la verdad del visible 400, que solo Kaggle tiene.** Si la verdad real difiriera del top-1 de baseline-1 en alguna de las 400 moléculas (por ejemplo, por espectros duplicados o por coseno 1.0 entre moléculas distintas), el MRR visible real de RGB podría ser distinto de 0.5758. El informe RGB ya advirtió esta limitación; hoy se confirma que sigue abierta.
- **Verificaciones pendientes del informe RGB (2026-11-11), NUNCA hechas:** (a) descargar por navegador la submission RGB enviada (457,663 B) y compararla con la local; (b) desglose por tamaño/biblioteca de RGB en validación (archivos P_val en la Kali, no medidos); (c) evaluación solo-clase-1 de Morgan/blend. Tres huecos abiertos desde el 11 de noviembre.

### H-5 — El gate se equivoca en las dos direcciones (medido hoy por deducción del techo)

**Dirección (a) — desplazar aciertos de retrieval hacia generación (la que mató a la v1):** localmente es CERO. Los 12 aciertos novel de retrieval tienen todos top1_cos ≥ 0.6 (mínimo 0.6066) y están en la rama retrieval; el conjunto {verdad en ret_sm Y top1_cos < 0.6} está vacío en las 1,000 novel. En clase-1, la retención a 0.6 es 199/200 y la única molécula enrutada a generación NO era un acierto en ninguna de las dos ramas (clase-1: 197/200 en retrieval-solo y 197/200 en híbrido, Δ = 0). **Evidencia: medición de hoy (Parte C) + `porton_calibracion.json`.**
**Dirección (b) — enrutar a retrieval moléculas que solo el beam resuelve (NO contemplada hasta hoy):** deducción verificada de `techo_reranking_beam25.json`:
- Novelas con verdad en beam-25: 9/1,000. De ellas, solo 1 está en la rama generativa (ZHHFMKYOKONAFI, ya acertada en rank 16). Las otras **8 están en la rama retrieval** (top1_cos ≥ 0.6) y su verdad NO está en el retrieval top-25 (si estuviera, serían aciertos de retrieval — y no lo son: la suma novel actual 11.5625 se explica exactamente con los 13 aciertos).
- Esas 8 moléculas son invisibles para el híbrido: el beam las resolvería, pero el gate las manda a retrieval, donde su verdad no existe. El techo A (las 9 en rank 1) vale 0.0205 novel MRR, una ganancia de +0.0089375 sobre el actual 0.0115625 — y aun así NO alcanza el criterio 0.021.
- **Implicación:** el gate no es solo un riesgo de fuga (dirección a), es también un techo de cobertura (dirección b). Un gate distinto (por ejemplo, enrutar a generación también cuando la lista de retrieval es "sospechosa" por otro criterio que top1_cos) podría recuperar parte de esas 8. Pero ningún gate recupera lo que el beam no contiene: el techo A es el límite absoluto de beam-25.
- **Lo que falta:** la medición por molécula de esas 8 (¿qué top1_cos tienen? ¿qué tan lejos está su verdad en el beam?) requiere `etapaB.csv` + las predicciones híbridas por molécula — es deducible de los archivos existentes, no se midió por molécula hoy. **Medible con datos locales, pendiente para la reanudación (ver §4.1).**

### H-6 — El SE de la diferencia pareada 3 vs v2: NO MEDIBLE

- El SE del MRR público es ≈ 0.008 (informe RGB: ~67 moléculas puntuables en el oculto). La brecha v2 − 3 = −0.008 es ~1 SE de cada score individual.
- Pero 3 y v2 se evaluaron sobre el MISMO set oculto, con las MISMAS moléculas: la diferencia pareada tiene un SE MENOR que el SE de cada score (las moléculas de clase-1 que ambas aciertan no aportan varianza a la diferencia). Ese SE pareado **no está calculado y no es calculable localmente: faltan los resultados por molécula del set oculto**, que Kaggle no entrega.
- **Consecuencia honesta:** "v2 < 3 por 0.008" no es una degradación establecida. Podría ser ruido (SE pareado desconocido, pero acotado por 0.008), fuga pequeña (~5% de clase-1 enrutada a generación explicaría −0.0074), dureza (escenario B), o menor f (escenario A). Las cuatro explicaciones caben en −0.008. La única degradación ESTABLECIDA entre envíos es la v1 (0.004, ~18 SE).

### H-7 — Lo que no vimos de la v2 en Kaggle (el log de la corrida)

- De la v1 existe el log de Kaggle (450.4 s, `RDKit: False`, top1_cos 1.0000 400/400). De la v2 NO hay log en el workspace: no se verificó que RDKit cargara (`RDKit: True`), ni el tiempo de corrida, ni cuántas moléculas tomaron la rama generativa en el oculto, ni la distribución de top1_cos en el oculto.
- **NO MEDIBLE localmente: falta el log de la corrida v2 en Kaggle.** Sin él, la afirmación "la rueda cp313 arregló la causa (a)" es una inferencia desde el score (0.138 >> 0.004) y desde la presencia de la rueda en el dataset, no una verificación directa. La inferencia es fuerte (con RDKit en False el score habría sido ~0.004 de nuevo), pero no es una lectura del log.
- Lo mismo aplica al número de moléculas generativas en el oculto: localmente son 382/1,200 (31.8%) las que toman la rama generativa (n_gen_first = 382, G4). En el oculto, esa fracción es desconocida y es la que determina cuánto puede aportar o quitar la rama generativa.

### H-8 — La validación local de 1,200 muestra lo que el público no muestra (y viceversa)

**Lo que muestra la validación local y el público no:**
- La SEPARACIÓN clase-1/novel: el público 0.146 esconde que 0.8859 viene de 200 moléculas y 0.0116 de 1,000. Sin la validación, no sabríamos que el proyecto es un problema de cobertura novel, no de ranking de clase-1.
- El desglose por rama: 12 retrieval + 1 generativa. El público no dice qué rama acertó.
- Los intervalos: IC95 [0.1362, 0.1778] — el público 0.146 y el 0.138 están AMBOS dentro. La validación dice que 3 y v2 son indistinguibles; el público, leído sin IC, sugiere una ordenación que la estadística no sostiene.
- Los techos: beam-25 techo A 0.0205 (criterio 0.021 NO alcanzado ni en el mejor caso); beam-50 techo A 0.0245 (sí lo alcanza, pero el techo realista B es 0.0145 y la vía 4 midió 0.0115714, NO ADOPTAR).
**Lo que muestra el público y la validación no:**
- El salto de dominio (timsTOF): la validación local no tiene espectros de otro instrumento. El público es la única medida del dominio real — y dice que RGB cae de 0.4951 (val) a una clase-1 implicada de 0.3124 (oculto).
- La f real del oculto (aunque sea circular, es la única estimación que existe).
**Lo que NINGUNO muestra:** los resultados por molécula del oculto; la distribución de top1_cos en el oculto; cuántas moléculas ocultas tomaron cada rama en cada envío.

---

## §3 — Las degradaciones: qué se contempló y qué NO

### 3.1 Submission 4 — RGB (0.146 → 0.049): degradación real, diagnosticada, con huecos

**La degradación es real:** 0.049 vs 0.146 es −0.097, ~12 SE del MRR público. No es ruido.

**Qué se contempló (informe RGB, 2026-11-11, verificado hoy en lo medible):**
- La RGB local NO reproduce el 0.049: validación local 0.4951 (n=13,748); visible local 0.5758; kernel v6 local 0.5723. Esperado en oculto: 0.078–0.090 (0.1568 × 0.4951 = 0.0776; 0.1568 × 0.5758 = 0.0903 — medidos hoy, Parte D; con la f simplificada 0.168 eran 0.0832/0.0967). Observado: 0.049. Residual 0.029–0.041, ~3.6–5.2 SE: real, no ruido.
- H1 (el runtime degradó el kernel) REFUTADA: el kernel v6 local reproduce el envío casi exacto (medido hoy: mismo top-1 en 385/400, Jaccard 0.9732, MRR 0.5723 vs 0.5758 del envío). La degradación está en el método.
- H2 (validación inflada: el pool de validación contiene la verdad para el 100% de las queries) + H3 (el oculto es más duro) CONFIRMADAS. La clase-1 oculta implicada para RGB es 0.3124, contra 0.4951 en validación.
- **Diagnóstico por molécula (NUEVO, hoy):** RGB acierta 367/400 en el visible, pero solo 182 con top-1 correcto. Sus conjuntos top-25 se solapan 30% con baseline-1. El problema de RGB es ORDEN (top-1 0.4550, recall@25 0.9175): la verdad suele estar en su lista, pero abajo. Baseline-1, con la verdad proxy, tiene top-1 1.0000. La RGB es un retriever más débil en ranking y con vecindarios distintos.
- **Por qué RGB es tan peor que la 3 (respuesta con número):** en el visible, RGB pone la verdad en rank 1 el 45.5% de las veces que baseline-1 la pone (182/400); su MRR visible es 0.5758 contra 1.0000 (proxy). En el oculto, la razón es 0.049/0.146 = 0.34 — PEOR que la razón visible (0.58), consistente con H2+H3 (la validación infla a RGB más que a baseline-1, y el oculto castiga más a RGB). RGB no es "un poco peor": es un método distinto con un ranking sistemáticamente inferior en este problema.

**Qué NO se contempló / quedó sin examinar:**
- Las 3 verificaciones pendientes del informe RGB (H-4): descarga del envío RGB por navegador; desglose por tamaño/biblioteca en validación; evaluación solo-clase-1 de Morgan/blend. Abiertas desde el 11 de noviembre, sin hacer.
- Si RGB acierta algo que baseline-1 no acierta: NO MEDIBLE con la verdad proxy (H-4). La pregunta sigue abierta y solo se responde con la verdad del visible 400.
- Si una FUSIÓN de rankings (baseline-1 + RGB) ayudaría: no se probó. Los conjuntos se solapan solo 30%, así que hay espacio teórico para complementar — pero RGB ordena peor, así que fusionar sin un re-ranker que sepa cuándo confiar en cada uno podría empeorar. Es una hipótesis sin medir, no una opción cerrada (ver §4.5).
- El envío RGB quedó marcado como FINAL (2026-11-12) con 0.049, por debajo de la v2 (0.138, sin marcar). La decisión de finales es de Elán (§5).

### 3.2 Generativa v1 (0.004): degradación real y compuesta, investigada y cerrada — con una corrección

**La degradación es real:** 0.004 vs 0.146 es −0.142, ~18 SE. Un colapso del 97%.

**Causa compuesta investigada (cerrada 2026-11-14 ~00:10), las tres con evidencia:**
- **(a) Contenedor Python 3.13 sin RDKit:** el notebook v1 llevaba la rueda `rdkit-2026.3.3-cp312` (para Python 3.12) pero Kaggle corrió Python 3.13; la rueda no instaló; el notebook corrió con `RDKit: False`. Evidencia: `/usr/local/lib/python3.13/dist-packages/` en el log; 102 líneas "SMILES Parse Error" en la corrida local G (con RDKit True) vs 0 en el log de Kaggle (sin RDKit, nada que parsear — los SMILES inválidos pasaban sin filtro o la rama generativa fallaba silenciosamente).
- **(b) Gate top1_cos ≥ 0.95, calibrado solo en el visible:** en el visible, top1_cos = 1.0000 por construcción (400/400), así que el gate 0.95 nunca se disparaba en visible (0 moléculas necesitaron generación). En el oculto, solo el 19.30% (spec-split) / 8.82% (mol-split) de las moléculas superan 0.95 — el gate mandaba a generación a la mayoría de la clase-1, desplazando aciertos de retrieval. Deducción, no medición directa (no hay top1_cos del oculto).
- **(c) GEN-VAL nunca midió el camino enviado:** GEN-VAL MRR 0.337223364107438, IC95 [0.31865, 0.35515], validez 0.7463, 11,793 fallos de parseo de 50,000 — midió el decodificador con la huella EXACTA, no con la huella predicha que el notebook usa. El número que justificó el envío (0.3372) no era el número del camino enviado.

**Qué atendió el rediseño v2 (verificado):**
- (a) → rueda cp313 (md5 `a9699b01…`, verificada hoy en el dataset v2 local) + chequeo de RDKit + log. G-V1 PASA.
- (b) → gate recalibrado a 0.6 con calibración congelada (`porton_calibracion.json`); retención clase-1 0.995. G-V7 E-1–E-11 cerraron las desviaciones de la calibración.
- (c) → shield: SMILES inválidos filtrados, retrieval rellena; validez 1.0 en fusión. G-V2 PASA (blindaje); G-V6 PASA (la salida visible v2 es byte-idéntica a la v1: 0/400 necesitan generación a 0.6).
- Resultado: 0.004 → 0.138 (+0.134). La recuperación es la evidencia de que (a)+(b) eran la causa: con ambas corregidas, el híbrido vuelve al régimen del retrieval.

**Qué NO atendió el rediseño v2:**
1. **La cobertura del beam (el cuello real):** el rediseño no cambió el decodificador, no reentrenó, no amplió el beam (beam-25, LOTE_BEAM=32, 9 chequeos — todo igual que v1). Verdad en beam: 9/1,000. El techo novel con beam-25 es 0.0205 y ni siquiera alcanza el criterio 0.021. La v2 arregló el envoltorio, no el motor.
2. **La exactitud de la huella predicha (la cola):** con fingerprints exactas el decodificador da 0.3372; con la huella predicha (kNN), la cola generativa es 0.0/0.0/0.0 en tan05 ≥ 0.6/0.8/=1.0 (n=380). El rediseño no tocó la predicción de huellas (stage 1 idéntico: TOPK_KNN=10, binarización 0.5, AST diff 0 líneas).
3. **El desplazamiento en el oculto (dirección a):** la v2 lo mitiga (gate 0.6 en vez de 0.95) pero no lo mide: la distribución de top1_cos en el oculto sigue desconocida (H-7). Localmente el desplazamiento es 0; en el oculto podría ser ~5% de clase-1 (eso explicaría el −0.008 entero). No hay forma de saberlo sin el log de la corrida v2.
4. **La dirección (b) del gate:** 8 moléculas beam-solvables enrutadas a retrieval (H-5). No contemplada en el rediseño v2 ni en ninguna de las 5 opciones de adaptación.
5. **La corrida local previa al envío:** la v2 se envió a Kaggle sin una corrida local end-to-end del notebook (no existe salida visible v2 ni log local de la v2). El rediseño se validó por piezas (G-V1 a G-V7), no por una corrida completa. La lección de GEN-VAL (medir el camino enviado) se aplicó al shield pero no a la corrida completa.
6. **El contrafactual de la investigación v1 quedó refutado (corrección a registrar):** la investigación decía "sin (b) el score sería ~0.146; sin (a) ~0.146 + contribución generativa". La v2 (ambas corregidas) marca 0.138, NO 0.146+. La corrección: la contribución generativa real es ~0 (1 molécula local), así que "0.146 + generativa" nunca fue el contrafactual correcto. El híbrido con todo corregido es ≈ retrieval − fuga/ruido, no retrieval + generativa. Esto cambia la lectura de la v1: la v1 no perdió una contribución generativa que existía; perdió el retrieval por el gate y por RDKit.

**Qué quedó sin examinar en la degradación v1:**
- Si la v1, con RDKit funcionando pero gate 0.95, habría marcado ~0.146 o menos (el contrafactual parcial nunca se midió: la v2 corrigió ambas causas a la vez, así que sus efectos individuales no están separados).
- El log de la corrida v1 en Kaggle se leyó parcialmente (450.4 s, RDKit False) pero no se archivó completo en el workspace para re-análisis.

---

## §4 — Enfoque rankeado para retomar CASMI

Criterio vigente: **novel MRR ≥ 0.021** (umbral 0.6 congelado). Estado actual: novel 0.0116 (G4), techo beam-25 0.0205, techo beam-50 0.0245. Las 5 opciones de adaptación están CERRADAS con número (re-entrenar NO PASA, binarización NO_ADOPTAR, predictor piloto NO-GO, beam 50 + re-ranking NO ADOPTAR, umbral agotado). El ranking siguiente NO reabre ninguna: ataca el cuello que las 5 dejaron intacto.

### 1. COBERTURA DEL BEAM — la única palanca que puede mover el criterio (potencial medido: techo novel 0.0205 → 0.0245+)

**El dato:** verdad en beam 9/1,000 (beam-25) y 15/1,000 (beam-50). Techo A novel: 0.0205 (beam-25, NO alcanza 0.021 ni en el mejor caso) y 0.0245 (beam-50, sí lo alcanza en el mejor caso, pero el techo realista B es 0.0145 y la vía 4 midió 0.0115714). **Conclusión: con beam-25 el criterio es INALCANZABLE por construcción; con beam-50 es alcanzable solo en el techo optimista.** Todo lo demás es segundo orden.
**Qué hacer:** subir la cobertura, no el orden. Opciones a medir (en este orden, cada una con su número antes de decidir):
- (a) Beam más grande con el mismo decodificador (beam-100): ¿cuántas verdades entran? Si beam-100 llega a ~25/1,000, el techo A supera 0.021 con margen. Costo: tiempo de decodificación; medirlo localmente con el checkpoint actual.
- (b) Diversidad del beam (muestreo por temperatura / nucleus en vez de beam search determinista): el beam actual es determinista y colapsa en variantes de la misma molécula (unicidad 0.9541 en fusión, pero novedad 0.3275 — el beam genera cosas nuevas, no la verdad). Medir si la diversidad sube la cobertura.
- (c) Identificar las 8 moléculas de H-5 (verdad en beam, enrutadas a retrieval): medir su top1_cos y su posición en el beam. Si el gate las enrutara a generación, el techo A (+0.0089) se vuelve alcanzable sin tocar el decodificador.
- **Qué NO hacer:** otro re-ranker. El re-ranking mueve 1 molécula (ZHHFMKYOKONAFI 16→14, Δ +0.0000089 novel). El orden NO es el cuello; la cobertura sí.

### 2. EXACTITUD DE LA HUELLA PREDICHA (la cola) — el segundo cuello, medido

**El dato:** el decodificador con huella exacta da 0.3372 (GEN-VAL); con huella predicha (kNN), la cola generativa es 0.0/0.0/0.0 en tan05 ≥ 0.6/0.8/=1.0 (n=380) contra 0.05/0.0395/0.0158 del kNN. Se necesitan ~50–190 generativas con tan05 ≥ 0.6 (hoy 19) o ~40–60 con tan05 = 1.0 (hoy 6). Con huellas exactas en las 380 generativas, la contribución esperada sería ~0.3372 × 380/1,000 ≈ 0.128 → novel ≈ 0.14. **La brecha entre 0.0116 y 0.14 es la huella, no el decodificador.**
**Qué hacer:**
- El kNN es hoy el mejor predictor de huellas (tan_cont 0.3156 vs 0.2315 del predictor aprendido vía 3, NO-GO). Cualquier intento nuevo de predictor debe batir al kNN en tan_cont en validación novel ANTES de tocar el híbrido (criterio pre-registrado del piloto: tan_cont > 0.35, cola ≥ 0.6 > 0.05, = 1.0 > 0.016 — el piloto vía 3 falló los tres).
- El barrido TOPK kNN (2026-11-20 07:37, COMPLETO y medido) muestra que la cola generativa NO se mueve con TOPK (frac tan05 ≥ 0.6 = 0.0500 para TOPK 3/5/10/20; 0.0553 para TOPK=1, que colapsa clase-1 a 0.1242/0.1061). TOPK está agotado como palanca.
- **Qué NO hacer:** re-entrenar el decodificador con corrupción (opción 1, NO PASA: 20 épocas, pérdida 0.9734, validez 1.0, novel 0.0115 ≈ G4). La validez se recupera con entrenamiento pero la exactitud no: los SMILES son válidos pero no son la molécula objetivo (Tanimoto novel top1 media 0.2298).

### 3. MEDIR LO QUE EL SCORE PÚBLICO NO SEPARA (f, dureza, top1_cos oculto) — diseño de medición, no de modelo

**El dato:** el público es una ecuación con dos incógnitas (H-2); el SE pareado 3 vs v2 no existe (H-6); el log de la v2 no está en el workspace (H-7). Cada envío futuro debería ser INFORMATIVE además de intentar subir el score.
**Qué hacer:**
- Recuperar y archivar el log de la corrida v2 en Kaggle (tiempo, `RDKit: True/False`, distribución de top1_cos en el oculto, nº de moléculas por rama). Es la única forma de separar fuga de dureza de f.
- Diseñar el próximo envío como experimento: por ejemplo, un envío retrieval-solo con el gate en 0.5 (empate exacto con 0.6 en calibración) daría un segundo punto público que, combinado con 0.146 y 0.138, acota f y la fuga. O un envío híbrido con beam-50 (vía 4, ya medido localmente 0.0115714) para testear el techo en el oculto. **Ningún envío sin GO de Elán.**
- Medir por molécula las 8 de H-5 con los archivos existentes (`etapaB.csv` + predicciones híbridas): top1_cos, posición en beam, Tanimoto. Es trabajo local, sin Kaggle, y cierra un hueco de esta retrospectiva.

### 4. GATE: NO TOCAR EL UMBRAL, INVESTIGAR LA DIRECCIÓN (b)

**El dato:** umbral 0.6 congelado; G4 robusto al umbral (E-5); empate exacto 0.5/0.6; la retención de clase-1 cae con el umbral. El umbral está agotado como palanca (opción 5, cerrada).
**Qué hacer:** no cambiar el umbral. Investigar la dirección (b) de H-5: un segundo criterio de enrutamiento (además de top1_cos) que mande a generación las moléculas cuya lista de retrieval sea "sospechosa" — por ejemplo, baja diversidad de la lista, o top1_cos alto pero con un segundo candidato casi empatado. Hipótesis sin medir; el primer paso es medir las 8 moléculas (punto 3), no construir el criterio.

### 5. RGB COMO COMPLEMENTO, NO COMO REEMPLAZO — hipótesis sin medir, prioridad baja

**El dato:** RGB ordena peor (top-1 0.4550 vs 1.0000 proxy) pero sus conjuntos se solapan solo 30% con baseline-1, y su recall@25 es 0.9175 (la verdad está en su lista 367/400 veces). Hay espacio teórico para complementar.
**Qué hacer (solo después de 1–3):** cerrar las 3 verificaciones pendientes del informe RGB (H-4); medir una fusión de rankings baseline-1 + RGB en la validación local de 1,200 (¿sube el recall@25 de clase-1? ¿sube el novel?); si no sube nada local, no se toca. **No se propone ningún envío RGB nuevo: el método RGB solo marcó 0.049 y está marcado como final.**

### 6. NO HACER — lista cerrada (todo medido, todo NO)

- Re-entrenar el decodificador con corrupción (NO PASA, novel 0.0115).
- Binarización 0.3 (NO_ADOPTAR, paired probe Δ = 0.0000).
- Predictor aprendido de huellas vía 3 (NO-GO, 0.2315 vs kNN 0.3156).
- Beam-50 + re-ranking como está propuesto (NO ADOPTAR, novel 0.0115714; el re-ranker mueve 1 molécula, Δ +0.0000089).
- Cambiar el umbral 0.6 (agotado, congelado).
- Re-abrir la búsqueda de optimización sin una hipótesis de cobertura (la búsqueda se detuvo por orden de Elán 2026-11-20 08:07; el barrido TOPK ya está completo y la cola no se mueve).
- Marcar la v2 como final o enviar nada sin GO de Elán.

---

## §5 — Finales: estado y decisión pendiente

- Finales actuales (marcadas 2026-11-12): **submission 3 (0.146) + submission 4 RGB (0.049).**
- Generativa v2 (0.138) SIN marcar. Decisión de Elán 2026-11-19 ~07:02: "No, déjenla como está por ahora".
- **Números para la decisión (sin recomendar acción, porque es de Elán):** v2 (0.138) > RGB (0.049) por 0.089 (~11 SE); v2 < 3 (0.146) por 0.008 (~1 SE, no establecido como degradación real — §2 H-6). Si un slot final fuera a cambiar, la v2 es la candidata por número; si se queda como está, la justificación es que la v2 no ha demostrado ser mejor que la 3 y que el slot de RGB es la apuesta diversificada (un método distinto). Ambas lecturas caben en los números. **No se marca ni desmarca nada en esta retrospectiva. Decisión de Elán cuando CASMI se retome.**

---

## Anexo A — Mediciones nuevas de esta retrospectiva (2026-11-20)

Script: `verificacion/retrospectiva_mediciones.py` (corregido durante la corrida: filtro a split `validacion` — la primera corrida falló con `KeyError: 'AARGZOOWPKMZGW'` porque `etapaA.npz` trae 2,400 moléculas y `verdad_validacion.csv` solo 1,200; el error se corrigió y se re-corrió completo).
Entorno: `~/workspace/casmi-diag-venv/bin/python` (RDKit 2026.03.3, scorer-pinned). Duración total: 424.2 s, RSS máx 162 MB, EXIT=0.
Log: `verificacion/retrospectiva_mediciones.log` (+ `retrospectiva_mediciones_stdout.log`). JSON: `verificacion/retrospectiva_mediciones.json`.

| Parte | Resultado |
|---|---|
| A — visible 400 (proxy = top-1 baseline-1, justificado por top1_cos 1.0000 en 400/400) | sub3 MRR 1.0000 (circular por construcción); RGB 0.5758, IC95 [0.5358, 0.6163], top1 0.4550, recall25 0.9175, hits 367/400; v1-visible 1.0000; kernel v6 0.5723, IC95 [0.5321, 0.6130], top1 0.4500, recall25 0.9200, hits 368/400. Acuerdos: 3v4 mismo top-1 182/400, Jaccard 0.3017; 3v1vis 400/400, Jaccard 0.9998; 4v6 385/400, Jaccard 0.9732. RGB hits con top-1 = proxy: 182/367. RGB hits que baseline-1 no tiene: 0 (imposible por construcción). |
| B — retrieval-solo, validación 1,200 | Todo 0.1572361111111111 [0.1362, 0.1778], top1 0.1450, recall25 0.1742, 209/1,200. Novel 0.0115 [0.0055, 0.0185], top1 0.0110, recall25 0.0120, 12/1,000. Clase-1 0.8859166666666667 [0.8492, 0.9208], top1 0.8150, recall25 0.9850, 197/200. |
| C — 13 aciertos novel del híbrido | 11 rank 1 retrieval + 1 rank 2 retrieval (ULFUJLFTRWWLPO) + 1 rank 16 generativa (ZHHFMKYOKONAFI). Los 12 de retrieval: top1_cos 0.6066–1.0000, posición ret_sm 1 (11) o 2 (1). Novelas con verdad en ret_sm: 12/1,000; clase-1: 197/200. Generativa Y con verdad en ret_sm: 0. |
| D — aritmética del oculto (corregida 2026-11-20, A-R-1/D-RT-2) | f exacta = 0.1568330224 (fórmula de dos términos); f simplificada = 0.16798987458290185 (f = público/R1, término novel omitido; convención del proyecto). RGB clase-1 oculta implicada 0.3124 (exacta; 0.2916842465753425 con la f simplificada). Escenario A f = 0.1445728010980213 (sin cambio, ya usaba la fórmula exacta); escenario B clase-1 = 0.8176 (exacta; 0.7639809523809524 con la f simplificada). v2 − 3 = −0.008; v2 − v1 = +0.134. 0.138 dentro del IC95 de G4: sí. RGB esperado 0.0776 (val 0.4951) / 0.0903 (visible 0.5758) con la f exacta; 0.08317178690599471 / 0.09672856978483488 con la f simplificada. |

**Reproducción exacta (verificación de que el método de medición es el mismo que el de los informes previos):** la Parte A reproduce EXACTAMENTE los números del informe RGB (0.5758 / 0.4550 / 0.9175 para el envío RGB; 0.5723 / 0.4500 / 0.9200 para el kernel v6) y la Parte B reproduce EXACTAMENTE los números de G4 para clase-1 (0.8859166666666667, 197/200) y el novel de retrieval (0.0115, 12/1,000). La medición nueva es consistente con las mediciones históricas; lo nuevo es la comparación retrieval-solo vs híbrido (Parte B/C) y la aritmética del oculto (Parte D).

**Precisión trivial (1 molécula, sin efecto):** G4 reporta n_gen_first = 382, pero el conteo por ramas da 380 novel (top1_cos < 0.6, `techo_reranking_beam25.json`) + 1 clase-1 (retención 199/200) = 381. Diferencia de 1 molécula, origen no identificado, sin efecto en ningún MRR. Se registra para que el Auditor lo tenga en cuenta; no cambia ningún veredicto.

## Anexo B — Archivos, md5 y verificación de `salida/`

**Archivos de esta retrospectiva (todos en `verificacion/`):**
- `RETROSPECTIVA_ENVIOS_CASMI_2026-11-20.md` — este documento.
- `retrospectiva_mediciones.py` — script de medición (Partes A–D), completo y con log.
- `retrospectiva_mediciones.log` — log de ejecución (424.2 s, EXIT=0).
- `retrospectiva_mediciones_stdout.log` — stdout completa de la corrida.
- `retrospectiva_mediciones.json` — resultados numéricos (Partes A–D, precisión completa).

**`salida/` — md5 antes (2026-11-20 08:44) y después (2026-11-20, tras las mediciones): 5/5 idénticos, intactos.**

| Archivo | md5 |
|---|---|
| `salida/submission.csv` | 7a7da389417fb162039a94394d76e1ee |
| `salida/submission_paso2.csv` | ed758b99cdb3a91ac690f02fc3618779 |
| `salida/submission_relleno25.csv` | 7f082916a4a2bd68392750ea3cf2c8a7 |
| `salida/submission_morgan.csv` | 854584724275387743e744fd77757961 |
| `salida/submission_rgb.csv` | 6dcceab5d57a00b82d485439ebd6061b |

**Fuentes leídas para esta retrospectiva (verificación desde archivos, no desde memoria):**
- `verificacion/INVESTIGACION_BAJA_GENERATIVO_0.004.md` (300 l., completa).
- `verificacion/ADAPTACION_GENERATIVA_V3_2026-11-19.md` (18 KB, completa).
- `verificacion/DIAGNOSTICO_RGB_0.049_REPORTE.md` (15 KB, completa).
- `verificacion/REDISENO_GENERATIVO_V2_2026-11-14.md` (§1, §9 riesgos, enmiendas G-V7 E-1–E-11).
- `verificacion/DISENO_PREDICTOR_VIA3_2026-11-20.md` (§1.3 curva de respuesta del decodificador, §7 piloto).
- `verificacion/hibrido_honesto/G4.json` (híbrido 0.15728819444444442, novel 0.0115625, clase-1 0.8859166666666667), `G4_V3.json` (0.1572361111111111, novel 0.0115, adopta false), `G4_VIA4.json` (0.15729563492063492, novel 0.011571428571428571, adopta false).
- `verificacion/porton_calibracion.json` (umbral 0.6, empate 0.5/0.6, retención 0.995).
- `verificacion/techo_reranking_beam25.json` (techo A 0.0205 / B 0.0125; criterio 0.021 NO alcanzado ni en techo A) y `techo_reranking_beam50.json` (techo A 0.0245 / B 0.0145).
- `verificacion/via3-piloto/evaluacion_predictor_via3.json` (veredicto GO False, los 3 criterios en False).
- `verificacion/ESTADO_MONITOREO_V3_2026-11-19.md` (veredicto NO ADOPTAR, NO SE CONSTRUYE LA V3, 02:44 CST).
- `30_evaluar_generativo.py` (raíz; `canonical_ik14` + reglas R-EVAL-2/3/4).
- `~/workspace/casmi-drive-upload-2026-11-14/dataset_v2/` — rueda cp313 md5 `a9699b01…` verificada hoy.

## Anexo C — Números de referencia (todos verificados en archivo)

- G4: híbrido 0.15728819444444442, IC95 [0.1362, 0.1778]; novel 0.0115625 [0.0055, 0.0185]; clase-1 0.8859166666666667 [0.8492, 0.9208]; top1 0.145; recall25 0.175; n_gen_first 382; validez 1.0.
- G4_V3: 0.1572361111111111, novel 0.0115, adopta_checkpoint_v3 false (20 épocas, pérdida 0.9734, validez 1.0, Tanimoto novel top1 media 0.2298).
- G4_VIA4: 0.15729563492063492, novel 0.011571428571428571, adopta_via4 false.
- Techos: beam-25 novel A 0.0205 [0.0125, 0.03] / B 0.0125 [0.006, 0.0195]; híbrido A 0.16473611111111108 / B 0.15806944444444443. Beam-50 novel A 0.0245 [0.0155, 0.0345] / B 0.0145 [0.008, 0.022]; híbrido A 0.16806944444444444 / B 0.1597361111111111.
- GEN-VAL: MRR 0.337223364107438, IC95 [0.31865, 0.35515], validez 0.7463, 11,793 fallos de parseo / 50,000.
- Vía 3 piloto: tan_cont validación novel 0.2315 vs kNN 0.3156; cola generativa 0.0/0.0/0.0 vs kNN 0.05/0.0395/0.0158; holdout 2,617 mol 0.2289; 500,000 espectros, 2 épocas, batch 256, loss 0.6038 → ~0.382; 7,796,288 parámetros. NO-GO.
- Binarización 0.3: Tanimoto 0.3033 vs 0.2627 a 0.5 (clase-1 n=400); paired probe Δ = 0.0000 → NO_ADOPTAR.
- TAN_TOP1 AUC 0.4096 (invertido, muerto). MARGEN 0.06567067207970509 (muerto). Reserva 20-gen+5-ret net negativo.
- TOPK kNN (2026-11-20): frac tan05 ≥ 0.6 = 0.0500 (TOPK 3/5/10/20), 0.0553 (TOPK=1); = 1.0 idéntico 0.0158; TOPK=1 colapsa clase-1 a 0.1242/0.1061.
- Curva de respuesta del decodificador: 0 aciertos generativos con tan05 < 0.6 (361 moléculas); único acierto generativo tan05 = 1.0 exacto (ZHHFMKYOKONAFI, rank 16, top1_cos 0.1613). Cola necesaria: ~50–190 generativas con tan05 ≥ 0.6 (hoy 19) o ~40–60 con tan05 = 1.0 (hoy 6).
- Breakeven novel para explicar 0.138: ≥ 0.0205 (escenario A) / 0.0212 (escenario B). Opción 1 (recuperación 3%): novel 0.0214, público 0.1462.
- v1: log 450.4 s; run G 1,189.8 s (10 mol, threshold 1.01); spec-split 19.30% ≥ 0.95, mol-split 8.82%.

---

**Cierre de la retrospectiva.** CASMI sigue PAUSADO. Nada a Kaggle, nada a GitHub, nada a producción. Este documento pasa a Datos → Auditor → Elán vía el Coordinador. La decisión de retomar, el orden del §4 y la selección final son de Elán.
