#!/usr/bin/env python3
"""CASMI — 30_evaluar_generativo.py: evaluador honesto para modelos
generativos de novo (protocolo PROTOCOLO_GENERATIVO_DE_NOVO.md).

Qué mide, sobre un archivo de predicciones por molécula
(molecule_id, smiles con hasta 25 SMILES separados por ";"):
- MRR@25 EXACTO contra la verdad (InChIKey14), con la MISMA
  canonicalización que el scorer oficial: MolFromSmiles ->
  TautomerEnumerator().Canonicalize -> MolToInchiKey -> primer bloque
  (código idéntico a verificacion/diagnostico_scorer.py, RDKit 2026.03.3,
  la versión pinned del scorer). Solo stdlib + numpy + RDKit: corre con
  ~/workspace/casmi-diag-venv/bin/python.
- Validez química, unicidad y novedad (contra --train-ik14, opcional).
- Cercanía estructural secundaria: Tanimoto@1 y mejor-en-25 (Morgan 2048,
  radio 2, igual que 04_huellas.py) si la verdad trae columna `smiles`.
- IC 95% del MRR@25 por bootstrap (semilla 42).

Reglas de evaluación (protocolo R-EVAL-1..4, ver el .md):
- R-EVAL-1: la verdad de cada molécula es su InChIKey14. La lista de
  predicción se procesa en orden, truncada a las primeras 25.
- R-EVAL-2: un SMILES inválido (falla parseo, tautómero o InChIKey)
  CONSUME su posición: nunca casa y no se salta.
- R-EVAL-3: los duplicados (mismo InChIKey14) colapsan a la primera
  aparición y NO consumen posición nueva.
- R-EVAL-4: el rango es la posición 1-based en la lista procesada;
  ausente = 0. MRR@25 = media de 1/rango (0 si ausente), idéntico a
  casmi_lib.mrr_from_ranks (reimplementado aquí porque casmi_lib
  necesita pandas/scipy y este script corre en el venv del scorer).

El manejo de duplicados/inválidos por parte del scorer OFICIAL no está
verificado (no hay acceso al código del scorer): R-EVAL-2/3 son la
regla CONSERVADORA del protocolo local. Si el scorer oficial fuera más
benévolo, el MRR local subestima; nunca sobrestima por esta vía.

Uso:
  ~/workspace/casmi-diag-venv/bin/python 30_evaluar_generativo.py \
    --predicciones <csv> --verdad <csv> --salida <json> \
    [--train-ik14 <txt>] [--semilla 42] [--bootstrap 2000]
  ~/workspace/casmi-diag-venv/bin/python 30_evaluar_generativo.py --autotest

--autotest: verifica la aritmética del evaluador con casos sintéticos
construidos desde intermedios/muestra_smiles_validacion_rdkit.txt
(500 SMILES del pool) y reproduce la observación medida: 8/500
moléculas cambian de InChIKey14 bajo canonicalización de tautómeros
frente a su inchikey14 registrado (ver protocolo §7). No entrena nada.
"""
import argparse
import csv
import json
import time
import resource
from pathlib import Path

import numpy as np

T0 = time.time()


def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time() - T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)


BASE = Path("/home/hatch/workspace/enveda-casmi-2026")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"
UMBRAL_CERCANO = 0.675
SEMILLA = 42


def _rdkit():
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    return Chem, rdMolStandardize


def canonical_ik14(smiles, chem, tautomer_enumerator):
    """SMILES -> (inchikey14, paso_de_fallo o '').

    Idéntico a verificacion/diagnostico_scorer.py: parseo, canonicalización
    de tautómeros, InChIKey, primer bloque.
    """
    try:
        mol = chem.MolFromSmiles(smiles)
    except Exception as exc:  # noqa: BLE001
        return "", f"parseo: {type(exc).__name__}"
    if mol is None:
        return "", "parseo: MolFromSmiles devolvió None"
    try:
        canon = tautomer_enumerator.Canonicalize(mol)
    except Exception as exc:  # noqa: BLE001
        return "", f"tautomero: {type(exc).__name__}"
    if canon is None:
        return "", "tautomero: Canonicalize devolvió None"
    try:
        key = chem.MolToInchiKey(canon)
    except Exception as exc:  # noqa: BLE001
        return "", f"inchikey: {type(exc).__name__}"
    if not key:
        return "", "inchikey: cadena vacía"
    return key.split("-")[0], ""


def mrr_from_ranks(ranks):
    """Idéntico a casmi_lib.mrr_from_ranks (misma fórmula)."""
    r = np.asarray(ranks, dtype=np.float64)
    rr = np.where(r > 0, 1.0 / np.where(r > 0, r, 1.0), 0.0)
    return {
        "n": int(len(r)),
        "mrr25": float(rr.mean()),
        "top1": float((r == 1).mean()),
        "top5": float(((r >= 1) & (r <= 5)).mean()),
        "recall25": float((r > 0).mean()),
        "rank_mediana_presentes": float(np.median(r[r > 0])) if (r > 0).any() else None,
    }


def procesar_lista(candidatos, verdad_ik14, chem, tautomer_enumerator, stats):
    """Procesa una lista de SMILES (máx 25) según R-EVAL-2/3.

    Devuelve (rango 1-based o 0, lista_procesada de ik14/None,
    smiles_validos_distintos en orden de aparición,
    n_validos_crudos, n_distintos).
    n_validos_crudos cuenta TODOS los SMILES crudos que parsean (incluidos
    duplicados); n_distintos, los InChIKey14 distintos. La validez es
    n_validos_crudos / n_crudos y la unicidad es n_distintos /
    n_validos_crudos: un duplicado baja la unicidad, no la validez.
    """
    stats["total_candidatos"] += len(candidatos)
    vistos = set()
    procesada = []
    smiles_validos = []
    n_validos_crudos = 0
    for smi in candidatos[:25]:
        if not smi:
            stats["vacios"] += 1
            procesada.append(None)
            continue
        ik14, fallo = canonical_ik14(smi, chem, tautomer_enumerator)
        if fallo:
            paso = fallo.split(":")[0]
            stats[f"fallo_{paso}"] = stats.get(f"fallo_{paso}", 0) + 1
            procesada.append(None)
            continue
        stats["ok"] += 1
        n_validos_crudos += 1
        if ik14 in vistos:
            stats["duplicados"] += 1
            continue
        vistos.add(ik14)
        procesada.append(ik14)
        smiles_validos.append(smi)
    rango = 0
    for i, ik14 in enumerate(procesada):
        if ik14 is not None and ik14 == verdad_ik14:
            rango = i + 1
            break
    return rango, procesada, smiles_validos, n_validos_crudos, len(vistos)


def evaluar(pred_rows, verdad, chem, tautomer_enumerator, train_ik14=None,
            semilla=SEMILLA, n_bootstrap=2000):
    """pred_rows: lista de dicts {molecule_id, smiles}.
    verdad: dict molecule_id -> {inchikey14, smiles|None}.
    """
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    stats = {"total_candidatos": 0, "ok": 0, "duplicados": 0, "vacios": 0}
    ranks = []
    filas_menos_25 = 0
    validez_por_fila = []
    unicidad_por_fila = []
    novedad_por_fila = []
    tanimoto_top1 = []
    tanimoto_mejor25 = []
    n_sin_verdad = 0
    for row in pred_rows:
        mid = row["molecule_id"]
        if mid not in verdad:
            n_sin_verdad += 1
            continue
        v = verdad[mid]
        cands = row["smiles"].split(";") if row["smiles"] else []
        if len(cands) < 25:
            filas_menos_25 += 1
        rango, procesada, smiles_validos, n_validos, n_distintos = procesar_lista(
            cands, v["inchikey14"], chem, tautomer_enumerator, stats)
        ranks.append(rango)
        validez_por_fila.append(n_validos / max(len(cands), 1))
        unicidad_por_fila.append(n_distintos / max(n_validos, 1) if n_validos else 1.0)
        if train_ik14 is not None and n_distintos:
            nuevos = sum(1 for x in procesada if x is not None and x not in train_ik14)
            novedad_por_fila.append(nuevos / n_distintos)
        # Cercanía estructural (solo si la verdad trae SMILES). Se mide
        # sobre los candidatos VÁLIDOS distintos, en orden: sims[0] es el
        # primer candidato válido (el top-1 si el top-1 es válido).
        if v.get("smiles"):
            mol_v = chem.MolFromSmiles(v["smiles"])
            if mol_v is not None:
                fp_v = gen.GetFingerprint(mol_v)
                bits_v = set(fp_v.GetOnBits())
                sims = []
                for smi in smiles_validos:
                    mol_c = chem.MolFromSmiles(smi)
                    if mol_c is None:
                        continue
                    bits_c = set(gen.GetFingerprint(mol_c).GetOnBits())
                    union = bits_c | bits_v
                    sims.append(len(bits_c & bits_v) / len(union) if union else 0.0)
                if sims:
                    tanimoto_top1.append(sims[0])
                    tanimoto_mejor25.append(max(sims))
    met = mrr_from_ranks(ranks)
    # IC 95% bootstrap del MRR@25.
    rng = np.random.default_rng(semilla)
    r = np.asarray(ranks, dtype=np.float64)
    rr = np.where(r > 0, 1.0 / np.where(r > 0, r, 1.0), 0.0)
    if len(rr) and n_bootstrap > 0:
        boots = rng.choice(rr, size=(n_bootstrap, len(rr)), replace=True).mean(axis=1)
        ic = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    else:
        ic = [None, None]
    out = dict(met)
    out.update({
        "mrr25_ic95_bootstrap": ic,
        "bootstrap_n": int(n_bootstrap),
        "semilla": int(semilla),
        "filas_evaluadas": int(len(ranks)),
        "filas_sin_verdad": int(n_sin_verdad),
        "filas_con_menos_de_25": int(filas_menos_25),
        "validez_media": float(np.mean(validez_por_fila)) if validez_por_fila else None,
        "unicidad_media": float(np.mean(unicidad_por_fila)) if unicidad_por_fila else None,
        "novedad_media": float(np.mean(novedad_por_fila)) if novedad_por_fila else None,
        "novedad_n": int(len(novedad_por_fila)),
        "fallos_candidatos": stats,
    })
    if tanimoto_top1:
        t1 = np.asarray(tanimoto_top1)
        bm = np.asarray(tanimoto_mejor25)
        out["tanimoto"] = {
            "n": int(len(t1)),
            "top1_media": float(t1.mean()),
            "top1_mediana": float(np.median(t1)),
            "top1_frac_ge_0.675": float((t1 >= UMBRAL_CERCANO).mean()),
            "mejor25_media": float(bm.mean()),
            "mejor25_mediana": float(np.median(bm)),
            "mejor25_frac_ge_0.675": float((bm >= UMBRAL_CERCANO).mean()),
            "umbral_cercano": UMBRAL_CERCANO,
        }
    else:
        out["tanimoto"] = None
    return out


def leer_verdad(path):
    verdad = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            verdad[row["molecule_id"]] = {
                "inchikey14": row["inchikey14"].strip(),
                "smiles": (row.get("smiles") or "").strip() or None,
            }
    return verdad


def leer_predicciones(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def autotest():
    """Casos sintéticos con resultado exacto conocido + observación del
    tautómero (8/500). Falla (assert) si la aritmética no coincide."""
    chem, rdm = _rdkit()
    te = rdm.TautomerEnumerator()
    log("AUTOTEST: muestra de 500 SMILES del pool (stride 551)")
    sms = [l.strip() for l in (INTER / "muestra_smiles_validacion_rdkit.txt").read_text().splitlines() if l.strip()]
    iks_csv = [l.strip() for l in (INTER / "muestra_ik14_validacion_rdkit.csv").read_text().splitlines()[1:] if l.strip()]
    assert len(sms) == 500 and len(iks_csv) == 500, (len(sms), len(iks_csv))
    # Observación medida: tautómero vs directo contra el inchikey14 registrado.
    n_tau_difiere = 0
    for smi, ik_csv in zip(sms, iks_csv):
        mol = chem.MolFromSmiles(smi)
        k_dir = chem.MolToInchiKey(mol).split("-")[0]
        assert k_dir == ik_csv, "el inchikey14 registrado no es el directo: el pareo no es posicional"
        k_tau = chem.MolToInchiKey(te.Canonicalize(mol)).split("-")[0]
        if k_tau != ik_csv:
            n_tau_difiere += 1
    log(f"OBSERVACIÓN (medida): {n_tau_difiere}/500 SMILES cambian de "
        f"InChIKey14 bajo canonicalización de tautómeros (1.6%)")
    # Verdad autoconsistente para el autotest: la clave tautomerizada del
    # propio SMILES (el evaluador debe casar lo que el scorer casaría).
    verdad = {}
    for i, smi in enumerate(sms):
        ik14, fallo = canonical_ik14(smi, chem, te)
        assert not fallo, (i, fallo)
        verdad[f"mol_{i}"] = {"inchikey14": ik14, "smiles": smi}
    n = 100  # subconjunto para los casos
    distractor = sms[200:225]  # 25 SMILES distintos, ninguno es verdad de mol_0..99
    assert all(canonical_ik14(s, chem, te)[0] not in
               {verdad[f"mol_{i}"]["inchikey14"] for i in range(n)} for s in distractor)
    casos = {}
    # (filas, mrr_esp, top1_esp, rec_esp, validez_esp, unicidad_esp)
    # A: verdad en rango 1.
    rows_a = [{"molecule_id": f"mol_{i}",
               "smiles": ";".join([sms[i]] + distractor[:24])} for i in range(n)]
    casos["A_verdad_rank1"] = (rows_a, 1.0, 1.0, 1.0, 1.0, 1.0)
    # B: verdad en rango 5.
    rows_b = [{"molecule_id": f"mol_{i}",
               "smiles": ";".join(distractor[:4] + [sms[i]] + distractor[4:24])} for i in range(n)]
    casos["B_verdad_rank5"] = (rows_b, 0.2, 0.0, 1.0, 1.0, 1.0)
    # C: verdad ausente.
    rows_c = [{"molecule_id": f"mol_{i}", "smiles": ";".join(distractor)} for i in range(n)]
    casos["C_verdad_ausente"] = (rows_c, 0.0, 0.0, 0.0, 1.0, 1.0)
    # D: inválido en rango 1, verdad en rango 2 crudo -> rango 2 procesado.
    rows_d = [{"molecule_id": f"mol_{i}",
               "smiles": ";".join(["NO_ES_SMILES", sms[i]] + distractor[:23])} for i in range(n)]
    casos["D_invalido_rank1_verdad_rank2"] = (rows_d, 0.5, 0.0, 1.0, 0.96, 1.0)
    # E: duplicado de la verdad (rango 1, repetida) + 24 distractores.
    rows_e = [{"molecule_id": f"mol_{i}",
               "smiles": ";".join([sms[i], sms[i]] + distractor[:23])} for i in range(n)]
    casos["E_verdad_duplicada"] = (rows_e, 1.0, 1.0, 1.0, 1.0, 0.96)
    # F: 25 veces el mismo distractor (colapsa a 1), verdad ausente.
    rows_f = [{"molecule_id": f"mol_{i}", "smiles": ";".join([distractor[0]] * 25)} for i in range(n)]
    casos["F_distractor_x25"] = (rows_f, 0.0, 0.0, 0.0, 1.0, 0.04)
    resultados = {}
    for nombre, (rows, mrr_esp, top1_esp, rec_esp, val_esp, uni_esp) in casos.items():
        res = evaluar(rows, verdad, chem, te, n_bootstrap=0)
        assert abs(res["mrr25"] - mrr_esp) < 1e-9, (nombre, res["mrr25"], mrr_esp)
        assert abs(res["top1"] - top1_esp) < 1e-9, (nombre, res["top1"], top1_esp)
        assert abs(res["recall25"] - rec_esp) < 1e-9, (nombre, res["recall25"], rec_esp)
        assert abs(res["validez_media"] - val_esp) < 1e-9, (nombre, res["validez_media"], val_esp)
        assert abs(res["unicidad_media"] - uni_esp) < 1e-9, (nombre, res["unicidad_media"], uni_esp)
        resultados[nombre] = {"mrr25": res["mrr25"], "top1": res["top1"],
                              "recall25": res["recall25"],
                              "validez_media": res["validez_media"],
                              "unicidad_media": res["unicidad_media"]}
        log(f"AUTOTEST {nombre}: MRR {res['mrr25']:.4f} (esperado {mrr_esp}) "
            f"top1 {res['top1']:.4f} recall {res['recall25']:.4f} "
            f"validez {res['validez_media']:.3f} unicidad {res['unicidad_media']:.3f} — OK")
    # Tanimoto de control: caso A, top-1 es la verdad -> Tanimoto@1 = 1.0.
    res_a = evaluar(rows_a, verdad, chem, te, n_bootstrap=0)
    assert res_a["tanimoto"] is not None
    assert abs(res_a["tanimoto"]["top1_media"] - 1.0) < 1e-9, res_a["tanimoto"]
    log(f"AUTOTEST tanimoto caso A: top1_media {res_a['tanimoto']['top1_media']:.4f} "
        f"(esperado 1.0) — OK")
    out = {"autotest": "PASA", "n_por_caso": n,
           "tautomero_cambia_ik14": f"{n_tau_difiere}/500",
           "casos": resultados}
    (VERIF / "evaluador_generativo_autotest.json").write_text(json.dumps(out, indent=2))
    log("AUTOTEST COMPLETO: 6 casos + tanimoto, todos PASA")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predicciones", default="")
    ap.add_argument("--verdad", default="")
    ap.add_argument("--train-ik14", default="")
    ap.add_argument("--salida", default="")
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--autotest", action="store_true")
    args = ap.parse_args()
    if args.autotest:
        out = autotest()
        print(json.dumps(out, indent=2))
        return
    assert args.predicciones and args.verdad and args.salida, \
        "se requieren --predicciones, --verdad y --salida (o --autotest)"
    chem, rdm = _rdkit()
    te = rdm.TautomerEnumerator()
    log(f"RDKit {__import__('rdkit').__version__} (scorer pinned 2026.03.3)")
    verdad = leer_verdad(args.verdad)
    log(f"verdad: {len(verdad):,} moléculas ({args.verdad})")
    pred = leer_predicciones(args.predicciones)
    log(f"predicciones: {len(pred):,} filas ({args.predicciones})")
    train_ik14 = None
    if args.train_ik14:
        train_ik14 = {l.strip() for l in Path(args.train_ik14).read_text().splitlines() if l.strip()}
        log(f"train-ik14 para novedad: {len(train_ik14):,} ({args.train_ik14})")
    res = evaluar(pred, verdad, chem, te, train_ik14=train_ik14,
                  semilla=args.semilla, n_bootstrap=args.bootstrap)
    Path(args.salida).write_text(json.dumps(res, indent=2))
    log(f"RESULTADO: {json.dumps(res, indent=2)}")
    log(f"escrito {args.salida}")
    log("FIN — evaluador generativo")


if __name__ == "__main__":
    main()
