# CASMI — Evidencia del envío 3 (notebook en runtime)

Archivado 2026-11-09 ~20:55 CST a petición del Coordinador (auditoría
interna H-4: el envío 3 estaba reportado sin evidencia local archivada;
el informe runtime aún decía "pendiente de envío"). La evidencia es la
página de Kaggle leída con la sesión de Elán (elnzainoscorona) en dos
verificaciones de solo lectura: la del envío (2026-11-09 19:48–20:22 CST)
y la de confirmación (2026-11-09 ~20:55 CST). Nada se envió ni se cambió
en la segunda verificación.

## Notebook en Kaggle

- URL: https://www.kaggle.com/code/elnzainoscorona/casmi-runtime-notebook
- Título: casmi_runtime_notebook — autor Elán Zainos Corona, privado.
- Concurso vinculado: "Competition Notebook — Enveda CASMI 2026 -
  Molecule ID From Mass Spectra".
- Versión actual: Version 3 of 3 (última edición lun 9 nov 2026
  19:54:49 GMT-0600). Internet desactivado en los ajustes (verificado:
  el menú decía "Turn on internet").
- Public Score mostrado en el notebook: 0.146; Best Score: 0.146 V3.
- Runtime en Kaggle: 7m 12s (431.8 s); lenguaje Python.
- Código sin modificar respecto a `casmi_runtime_notebook.ipynb` local.
- Salida de la corrida: submission.csv (400.18 kB); logs: test recibido
  1,213 espectros, 400 moléculas; 400 filas, exactamente 25 candidatos
  por fila (10,000 total); todos los checks OK; md5
  `7f3fafa4ae01da0524ef5b18d8844ae0` (distinto al md5 local 7f082916…,
  lo que confirma que corrió sobre el test oculto, no el público).

## Envío en la página de submissions

- URL: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/submissions
- "casmi_runtime_notebook - Version 3" — Status: **Succeeded**;
  Public Score: **0.146**; enviado lun 9 nov 2026 20:04:39 GMT-0600
  (hora central). El enlace del envío abre el notebook con
  `?scriptVersionId=367881767`, que confirma el scriptVersionId
  **367881767**.
- Envíos diarios: el diálogo de envío reportó 4 restantes ANTES de este
  envío, así que tras él quedan 2 de 5 (la página de submissions no
  muestra el conteo restante).
- Finales: 0/2 seleccionados; nada marcado como final.
- Los 2 envíos anteriores siguen listados como "Submission Scoring
  Error" sin score ("—"): uno 19:26:58 GMT-0600 (scriptVersionId
  367876111) y otro 17:16:56 GMT-0600 (scriptVersionId 367850937).

## Archivos locales que lo respaldan

- `casmi_runtime_notebook.ipynb` (7 celdas, 5 de código, autocontenido).
- `kaggle-casmi-informe-runtime.md` (método + verificación local +
  sección de envío actualizada).
- `verificacion/runtime.json` (verificación local: salida byte por byte
  idéntica a `salida/submission_relleno25.csv`, md5 7f082916…, 400 filas,
  25 exactos, 10,000 SMILES válidos; ejecución local 313.1 s, RSS
  2,834 MB).
- `verificacion/runtime_check/submission.csv` (salida de la verificación
  local, md5 7f082916…).
