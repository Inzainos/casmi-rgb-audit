# CASMI 2026 — revisión del piloto B

## Decisión

El piloto B **no reemplaza el envío ni el experimento RGB**. Se conserva
como línea experimental porque aún no hay evidencia comparable de que mejore
el resultado público y no supera la puerta de calidad que el propio piloto
propuso. No se debe presentar como modelo listo para Kaggle.

## Comparación y alcance

| Experimento | Medida disponible | Alcance |
|---|---:|---|
| RGB bin | MRR@25 local 0.4951; público 0.049 | Retrieval desde espectro; el pool no contiene la mayoría de las moléculas novedosas |
| Piloto B | MRR@25 0.3372 (IC95% 0.3190–0.3551) | 2,000 GEN-VAL con huellas exactas como entrada; solo mide huella→SMILES |

Estas cifras **no son comparables directamente**: el piloto B no recibe
espectros ni huellas predichas. Además, su MRR de novedad no se midió
(`novedad_media: null`), por lo que tampoco se puede verificar el umbral
de go-ahead `MRR_novel > 0.01`.

## Puertas del piloto

- Holdout interno (2,000 evaluados): exact-match greedy **0.232** y
  exact-match beam-25 **0.000**, frente al umbral de corte propuesto de
  0.50. Beam-25 tuvo validez 0.8235.
- GEN-VAL con huella exacta: MRR@25 **0.3372**, top-1 Tanimoto **0.6373**,
  validez media **0.7467** y unicidad media **0.9679**. Son medidas del
  decodificador aislado, no del pipeline de envío.
- La auditoría no encontró solapamiento directo train/GEN-VAL ni train/spec,
  pero sí 152 y 81 solapamientos, respectivamente, después de canonicalizar
  tautómeros. El baseline empírico de memorizar train fue MRR@25 **0.0019**.
  La auditoría original marca `pasa: false` por ese solapamiento; debe
  conservarse como limitación al interpretar las métricas.

## Qué falta para considerarlo para envío

1. Entrenar la etapa 1 espectro→huella y evaluar la integración completa con
   la etapa 2 huella→SMILES en un split que no incluya la verdad en el pool.
2. Medir `MRR_novel` contra el train real y confirmar el scoring con la misma
   canonicalización de la competencia.
3. Repetir la evaluación de forma independiente y comparar el resultado
   público antes de cambiar el envío actual.

## Adaptación del decodificador a las salidas RGB

Sí es posible conectar las métricas: RGB produce 2,048 scores por molécula
y la entrada del decodificador es una huella Morgan binaria de 2,048 bits.
La adaptación usa exactamente la transformación RGB `lineal_bin` ya
evaluada —recorte a `[0,1]` y activación de los 100 scores mayores— antes de
decodificar. La predicción RGB debe guardarse como NPZ con arrays
`molecule_id` (IDs directos) y `predicciones` (matriz `n × 2048`). El nuevo
parámetro `--predicciones-rgb` rechaza IDs duplicados/faltantes y evita
silenciosamente asociar una huella a otra molécula.
La inferencia procesa dos moléculas por lote por defecto (`--batch-predecir`);
en una GPU de 4 GB se requiere un lote de 1 para evitar agotamiento de memoria.

Para empaquetar las salidas existentes de la validación RGB en el entorno
que contiene sus artefactos:

```python
import numpy as np

inter = "/ruta/al/proyecto/intermedios"
verif = "/ruta/al/proyecto/verificacion"
scores = np.load(f"{inter}/P_val_rgb20k.npy")
codes = np.load(f"{inter}/P_val_mol_rgb20k.npy")
ids = np.load(f"{verif}/mol_index.npy", allow_pickle=True)[codes].astype(str)
np.savez_compressed(
    f"{inter}/P_val_rgb20k_ids.npz",
    molecule_id=ids,
    predicciones=scores,
)
```

Después, con el checkpoint del piloto y el paquete de datos actualizados:

```sh
python codigo/piloto_b/entrenar_decodificador.py --modo predecir \
  --datos /ruta/al/paquete/datos \
  --salida /ruta/a/salida_rgb_piloto \
  --checkpoint /ruta/al/modelo.pt \
  --lista /ruta/al/paquete/datos/verificacion/piloto_b/gen_val_verdad.parquet \
  --predicciones-rgb /ruta/al/proyecto/intermedios/P_val_rgb20k_ids.npz \
  --device auto
```

Las predicciones SMILES resultantes se puntúan con `30_evaluar_generativo.py`
y la verdad del mismo split. Esto produce MRR@25, validez, unicidad y
Tanimoto de la cadena completa RGB→SMILES; no debe compararse con el
`MRR_novel` hasta pasarle también el conjunto train correcto al evaluador.

**Prueba de integración pendiente:** los dos ZIP recibidos no contienen
`W_lineal_rgb20k.npy`, los espectros bineados, las predicciones
`P_val_rgb20k.npy`/IDs ni el checkpoint del decodificador versionado. Por
tanto se probaron la binarización y el alineamiento con datos sintéticos,
pero no se pudo calcular aquí un nuevo MRR RGB→SMILES. Para cerrar esa
medición se necesitan esos artefactos del entorno de entrenamiento.

## Archivos versionados

- `codigo/piloto_b/`: entrenamiento, evaluación y notebook Colab limpiado.
- `validacion/piloto_b/`: métricas agregadas recibidas con el piloto.

No se versionaron el paquete de datos (39 MB), el checkpoint (34 MB) ni las
predicciones por molécula. El ZIP de datos incluido dentro del segundo ZIP es
idéntico al archivo `datos_piloto_b_colab.zip` recibido (SHA-256
`d36270b30f2b175660d968cdfa46980b33f877a93cdacc497284f5de6ed398f4`).
