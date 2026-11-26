# CASMI v4 — run completo 400 mol (Kali) — resultado recogido

Fecha de recogida: 2026-11-24 ~14:30 CST (tarea `casmi-v4-full-run-result`)
Run: PID 524430 en la Kali, trabajo `/home/deamon/casmi-sonda-v4/`, log `validacion_kali_v4_400mol.log`
Notebook: `casmi_notebook_generativo_v4_u01.ipynb` (md5 45248136…), 10 celdas

## Veredicto: TODO OK — 10/10 checks

- PID 524430 ya no corre; el run terminó 2026-11-24 14:11 CST.
- Tiempo total: 5736.1 s (~95.6 min). RSSmax 5342 MB. torch 2.15.0+cpu, RDKit: True, generativo: ACTIVO.
- Checks OK (10/10): columnas_exactas, filas_igual_a_moleculas_test, ids_unicos, ids_iguales_a_test,
  sin_nulos, sin_vacios, sin_candidatos_vacios, exactamente_25_por_fila, cero_filas_solo_CCO, blindaje_coherente.
- Sin FAIL / ERROR / Traceback en el log.
- `md5 submission.csv: 6184850794d4b50857c4889e32e53b88` (verificado por descarga directa con scp, coincide con el log).
- `submission escrita: 400 filas | retrieval-primero 400 | generativo-primero 0 | filas solo-CCO 0 | candidatos únicos/fila media 24.8`
- `validez beam (RDKit): 9507/10000 = 0.9507 (GEN-VAL medido 0.7463)`
- SMILES Parse Errors en el log (truncados tipo `O=C(Nc1c`, `Cc1cnc(C`): son candidatos beam inválidos descartados durante la generación, no un fallo del run.

## Comparación v4 vs v3 Test C (copia local md5 7f3fafa4ae01da0524ef5b18d8844ae0)

Formato: `molecule_id,smiles` con los 25 candidatos separados por `;` en el campo smiles.
- Header idéntico; 400 filas en ambas; ids iguales y en el mismo orden.
- Filas que difieren: 400/400.
- Posiciones 0–9 (ret top-10) idénticas: 392/400.
- Posiciones 10–24 (beam top-15) distintas: 400/400; 386/400 con las 15 posiciones distintas; media 14.8/15.
- Inyección V1 confirmada en las 400 filas.

### Las 8 filas con ret distinto — explicadas, no un fallo
`m_08d7f1, m_214040, m_997dbc, m_9f1fb6, m_b53b6e, m_b9f2b4, m_bcd97d, m_e5b5fa`
Son EXACTAMENTE las 8 filas donde la v3 sin-RDKit tenía moléculas duplicadas en su ret top-10
(37 huecos duplicados en total; p. ej. m_b53b6e tenía solo 2 únicas en ret). La v4 con RDKit
desduplica y rellena con moléculas distintas nuevas: cada molécula distinta del ret v3 sigue
presente en el ret v4 (`en v3 no en v4: []` en las 8). El prefijo coincide en 7/8
(m_b53b6e es la excepción porque solo tenía 2 únicas).

### Nota para el Sabueso (no bloquea los checks, que no miden unicidad del beam)
- 14 filas tienen duplicados DENTRO del beam (p. ej. m_b53b6e: 12 únicas de 25; m_08d7f1: 16/25).
  En esas 14 filas además 1 candidato beam coincide con el ret (ret∩beam = 1).
- El ret (0–9) de la v4 no tiene duplicados en ninguna fila (0/400), vs 8/400 en la v3 sin-RDKit.

## salida/ + portón — intactos, no tocados
- `~/workspace/enveda-casmi-2026/salida/` (en la raíz del proyecto, no dentro de `verificacion/`):
  5 archivos, mtimes 9–11 nov (sin tocar):
  submission.csv 7a7da389417fb162039a94394d76e1ee
  submission_morgan.csv 854584724275387743e744fd77757961
  submission_paso2.csv ed758b99cdb3a91ac690f02fc3618779
  submission_relleno25.csv 7f082916a4a2bd68392750ea3cf2c8a7
  submission_rgb.csv 6dcceab5d57a00b82d485439ebd6061b
- Portón `~/workspace/enveda-casmi-2026/verificacion/porton_calibracion.json`: md5 cacbe5660f7b532e498ce82e4f51cad9 — coincide con el esperado.

## Archivos
- Copia local de la submission v4: `hidden_files/casmi-v4-full-run/submission_v4_kali_400mol.csv` (md5 6184850794d4b50857c4889e32e53b88)
- Original en Kali: `/home/deamon/casmi-sonda-v4/verificacion/generativo_check/submission.csv` (375,534 B)

## Pendiente
- Verificación del Sabueso y relay al Coordinador. Nada enviado a Kaggle ni a GitHub.
  El envío va después, con GO de Elán. Finales sin tocar (v2 0.138 + sub3 0.146).
