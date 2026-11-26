# CASMI — Resultados E1 (beam-100) + E2 (beam condicionado) — 2026-11-20/21

GO de Elán 2026-11-20 13:27 ("Denle las dos propuestas"). Propuesta
`verificacion/PROPUESTA_OPTIMIZACION_CASMI_2026-11-20.md` (321 l., md5
`1070f5da…`, tras correcciones del Auditor A-POC-1/2/3 y de Datos
D-PO-4/D-PO-5). Cadena: Sabueso ✓ → Datos CONCURRO (PASA) → Auditor
CONCURRO (PASA) → Elán (GO). Nada a Kaggle, nada a GitHub, umbral 0.6
congelado, nada marcado como final. Cierre 14 dic 2026.

Valores CORREGIDOS usados en este informe (no los viejos): 975 novelas
sin verdad en ninguna lista (no 979); expectativa E2 ~2.1 aciertos
generativos (rango 1.9–2.35), NO ~6; ganancia novel estimada E2
+0.0007–0.0014 (no +0.002–0.004); ganancia máxima H-5 0.0010353 (no
0.001058); "100×" ≈ 67×.

## Resumen en una línea

- **E1 — CERRADA.** El techo de unión novel sube solo +0.0040 (0.0250 →
  0.0290), no material: IC95 solapados, las 4 moléculas nuevas están a
  rank 54–89 (fuera de toda fusión top-25), y ninguna fusión medida
  sobre beam-100 alcanza 0.021 (V0 0.0115, V1 máx 0.0121, vía 4 0.0116).
- **E2 — SIN VEREDICTO (bloqueado, no evaluable).** El beam murió a los
  672/1,200 por CUDA out of memory (lote 22); no hay CSV, no hay
  medición. El reintento a ciegas probablemente muera en el mismo
  lote (fallo posiblemente determinista); la decisión queda pendiente.

## E1 — Beam-100

### Criterio pre-registrado

La dirección sigue viva si el techo de unión (novel) sube
materialmente sobre 0.0250 (beam-50). Construir sobre beam-100 exige
además una fusión medida con novel ≥ 0.021.

### Costo real medido

- GPU (Kali, 1 worker, RTX 3050 Laptop): **39,948.7 s = 11.10 h**
  (lanzado 13:32 → FIN ~00:38 CST 21 nov). La estimación de la
  propuesta (~2 h) era extrapolación, no medición: el ritmo real fue
  368 s (lote 1) → ~1,300–1,650 s/lote (lotes 3–12) → ~1,900–2,760 s
  (lotes 13–17, con load 3–6 por watchdog/Firefox/npm/clamscan) →
  ~400–700 s/lote (lotes 19–35, load ~1.3). Media final 33.29 s/mol.
- Evaluación (local, CPU, casmi-diag-venv RDKit 2026.03.3):
  `evaluar_beam100.py` 360.0 s; `reconciliar_techo_beam100.py` 287.4 s;
  `evaluar_via4_beam100.py` 1,068.9 s; verificación V0 por molécula
  322.1 s. Total ≈ 2,038 s ≈ 34 min.

### Salidas (verificadas md5 Kali == local)

- `beam_via4_bin05_beam100.csv` md5 `cd9ec20c…` (1,201 l. = header +
  1,200; 2,867,789 B)
- `beam_via4_bin05_beam100_scores.csv` md5 `f29b11f7…` (120,001 l. =
  header + 120,000 scores; 7,395,375 B)
- `beam_via4_bin05_beam100.json` md5 `e1ab6cb4…` (checkpoint adoptado
  `2db2e48a…`, beam 100, 1,200 mol, candidatos 100/100/100, etapaA
  `aa61b8a1…`, tiempo 39,948.7 s)
- `beam_via4_bin05_beam100_gpu.log` md5 `6be16065…` (FIN 1,200 filas
  en 39,940 s)

### Números medidos

**Cobertura (novel, n=1,000).** Ranks FUSIONADOS (tras descartar
cadenas vacías, como hace la fusión); entre paréntesis los CRUDOS:

| Beam | Top-25 (MRR@25) | Lista completa |
|------|-----------------|----------------|
| beam-25 | 9 [3,4,7,11,12,15,16,18,25] | 9 |
| beam-50 | 8 [3,4,5,7,14,14,16,21] (crudos [3,4,5,8,15,15,17,22]) | 15 |
| beam-100 | **7 [3,4,5,11,17,18,19]** (crudos [3,4,5,12,18,19,20]) | **19** |

El beam-100 NO mejora la cobertura top-25 (7 < 8 < 9). En lista
completa conserva las 15 de beam-50 (perdidas: 0) y suma 4 nuevas,
todas profundas: NZOZNIDXTXBUDN (rank 76 crudo / 75 fus),
RDBPKCBIMTZUND (87/86), TUDRBRNMJFADST (54/53), VAXJJTLUQXIUOK
(89/88). Clase-1: beam-100 1/200 (rank 25 crudo / 24 fus), igual que
beam-50.

**Cadenas vacías (ranks crudos vs fusionados).** El CSV beam-100 trae
primer candidato vacío en **42 filas** (beam-50: 44) y 1,200 candidatos
vacíos en total (media 1/fila). Los ranks crudos y fusionados difieren
en 1 en esas filas (p. ej. ZHH: rank 30 crudo / 29 fusionado en
beam-100; en beam-50 era 22 crudo / 21 fusionado). Todos los ranks de
este informe declaran si son crudos o fusionados.

**Techo de unión ret ∪ beam → rank 1 (novel).** Dos definiciones, ambas
medidas con el mismo `ik14_de`:

| Beam | Unión ret DIRECTO ∪ beam | Unión ret CANÓNICO ∪ beam (definición de la propuesta) |
|------|--------------------------|--------------------------------------------------------|
| beam-25 | 17 (0.0170) | 21 (0.0210, IC95 [0.013,0.031]) |
| beam-50 | 22 (0.0220) | 25 (0.0250, IC95 [0.016,0.035]) |
| beam-100 | 26 (0.0260) | **29 (0.0290, IC95 [0.019,0.040])** |

La discrepancia aparente (evaluar_beam100.py mide 0.0220 donde la
propuesta registra 0.0250 para beam-50) quedó explicada y verificada
por lectura de código + recomputación (`reconciliar_techo_beam100.py`,
exit 0, ambos sanity PASS):
1. `rango_desde_ik14` solo mira los primeros 25 candidatos: la
   COBERTURA de evaluar_beam100.py es top-25 y no ve los aciertos del
   beam-50 en ranks 30–50 (7 moléculas). La propuesta usa pertenencia
   a conjunto sobre la lista completa.
2. La UNIÓN de evaluar_beam100.py usa ret_ik directo (8 aciertos
   novel); la propuesta usa ret canonicalizado vía `ik14_de(ret_sm)`
   (12 aciertos; el subconteo por tautomería es 4). Diferencia neta
   beam-50: 25 − 22 = 3.

**Techo de re-ranking beam-100 (novel):** actual 0.0115, A 0.0285
(IC95 [0.019,0.0395]), B 0.0155. Clase-1: actual = A = B = 0.8859.

**Fusiones medidas sobre beam-100 (novel MRR, umbral 0.6 congelado):**

| Fusión | Novel | Clase-1 | Híbrido | vs G4 (0.0115625) |
|--------|-------|---------|---------|-------------------|
| V0 crudo | 0.0115 | 0.8859 | 0.1572 | +0/−1 novel |
| V1 k=10 | 0.0118 | 0.8859 | 0.1575 | +2/−1 |
| V1 k=15 | 0.0119 | 0.8859 | 0.1576 | +3/−1 |
| V1 k=20 | 0.0121 | 0.8849 | 0.1576 | +6/−1 novel, −2 clase-1 |
| Vía 4 + re-ranker | **0.0116** | 0.8859 | 0.1573 | ≈ G4 (Δ −0.0000037) |

Ninguna alcanza 0.021. La vía 4 beam-100 (novel 0.0115588, híbrido
0.1572851, `G4_VIA4_BEAM100.json` md5 `be961226…`) es esencialmente
idéntica a G4 y micro-peor que vía 4 beam-50 (novel 0.0115714):
**NO ADOPTAR**.

La única molécula que empeora en V0 es **ZHHFMKYOKONAFI**, verificado
molécula a molécula: V0 beam-25 rank 16 → V0 beam-100 rank 0 (fuera
del top-25; en beam-100 está a rank 30 crudo / 29 fusionado).

### Veredicto E1: CERRADA

El techo de unión (definición de la propuesta) sube **+0.0040**
(0.0250 → 0.0290, 4 moléculas), que no es material: el IC95
[0.019,0.040] solapa ampliamente el de beam-50 [0.016,0.035]; la
propuesta esperaba ~35 verdades (techo ~0.035) y se midieron 29; las
4 nuevas están a rank 54–89, fuera del alcance de cualquier fusión
top-25 (el techo A de re-ranking es 0.0285 y la mejor fusión medida es
0.0121). La cobertura top-25 incluso baja (7 vs 8) y V0 pierde a ZHH.
No hay fusión medida que justifique construir sobre beam-100.

## E2 — Beam condicionado a ret_top1 (D-NEW-1)

### Criterio pre-registrado

VIVA si el beam condicionado mete la verdad en el beam de **≥ 10 de
las 120** novelas objetivo fuera del beam. Expectativa corregida por
el Auditor: ~2.1 aciertos generativos (rango 1.9–2.35), NO ~6
(proyección, no medición). Ganancia novel estimada corregida por
Datos (D-PO-5): +0.0007–0.0014.

### Costo real medido

- Entrada: ya construida y verificada antes del lanzamiento
  (`etapaA_e2.npz` md5 `52ac57d3…`, 42,625,916 B; manifest
  `c934f96d…`; CSV `a2331fb0…`; 300 modificadas, 228 filas B_test
  cambian, 72 bit-idénticas; objetivo 120: 86 modificadas, 34 no).
- GPU (Kali, 1 worker, beam 25): lanzado 01:16 21 nov → **muerto
  ~02:00 (44 min)**. Último lote completo 21/38 a 2,433.5 s
  (672/1,200). Ritmo: lotes 1–10 ~40 s, lotes 11–21 ~170–190 s.
  Total estimado si hubiera terminado: ~02:48 (~1.5 h, no ~33 min).

### Fallo (sin re-lanzamiento)

```
torch.AcceleratorError: CUDA error: out of memory
```

Traceback completo en `lanzar_e2_2026-11-21-0200.log` (66 l., md5 `58b11cf4…`).

## Adenda (Sabueso, 2026-11-21 ~02:26 — revisión del Auditor, CONCURRO PASA)

- **M-1 (corregida en la tabla de cobertura):** el rank fusionado de VGZKDIAIGIZOMF es 19 (beam-100) / 17 (beam-50), no 18/16. Causa: bug en `evaluar_beam100.json` (duplica el 19 crudo y el 18 fusionado; el JSON conserva el bug como artefacto registrado, md5 `e995e690…`); el informe copió esos valores. Listas correctas top-25: fusionados [3,4,5,11,17,18,19], crudos [3,4,5,12,18,19,20]. El conteo sigue 7, ningún MRR cambia.
- **M-2 (texto, no números):** `G4_VIA4_BEAM100.json` dice que el re-ranker recibe "los 50 candidatos del beam" — copy-paste rancio del doc beam-50. Verificado: `evaluar_via4_beam100.py` carga los 100 candidatos completos; los números de vía 4 (novel 0.0115588, híbrido 0.1572851) son sobre beam-100 completo.
- **Decisión del reintento de E2:** Elán dio el GO 2026-11-21 02:25 con las CUATRO medidas juntas — (a) lotes de 16, (b) salida por lote con reanudación, (c) aislar las moléculas del lote 22, (d) `empty_cache` entre lotes. Agente nuevo despachado 02:25; criterio pre-registrado sin cambio (dirección viva si ≥10 de las 120 novelas objetivo meten la verdad en el beam).
