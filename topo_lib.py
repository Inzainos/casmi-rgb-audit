"""CASMI paso 2 — topo_lib.py
Funciones del modelo topológico (adaptación del SNT Genomic Topologic
Analyzer, repo Inzainos/Genomic_agent, MIT; agente de dos niveles).

Mapeo (validado con los datos, no dado por bueno):
  hub = molécula; satélites = bits de huella de Morgan; observables = picos.
  R_b = fracción de los M picos satélite del bit b presentes en el espectro.
  Baseline por bit (librería de train, 1 espectro/molécula):
    μ+/σ+ de R_b sobre moléculas CON b; μ-/σ- sobre moléculas SIN b.
  Z = (R - μ)/σ (mismo umbral |Z| > 2.5 del agente genómico).
  Puntaje de presencia: score_b = Φ(Z_b^- - 2.5), con Z_b^- = (R_b - μ_b^-)/σ_b^-.
    score > 0.5  <=>  Z_b^- > 2.5  <=>  evidencia satélite anómalamente alta
    para una molécula SIN el bit  ->  LEAPFROG  ->  bit PRESENTE.
    Z_b^+ < -2.5 (evidencia muy por debajo de lo típico CON el bit)
    ->  SATELLITE_CAPTURE  ->  bit AUSENTE con confianza.
    Molécula sin ningún bit presente (n_present_L2 == 0) -> HUB_COLLAPSE
    (espectro degenerado; predicción no fiable) — bandera diagnóstica.

  v2 (binomial): la Z gaussiana con σ empírica está mal calibrada — algunas
  bits tienen σ^- diminuta (mín 0.0025) y cualquier R_b moderado les da
  Z enorme (precisión 13% al umbral 0.5 en v1; falsos positivos ~47/mol).
  La v2 usa σ_efectiva = máx(σ_empírica, σ_binomial), con
  σ_binomial = sqrt(p0·(1-p0)/n), p0 = μ^- (o μ^+), n = nº de picos del
  espectro consultado: es el test binomial de proporciones (la R_b es
  una fracción de conteo) con piso por sobredispersión empírica. La
  estructura Z = (R-μ)/σ y el umbral 2.5 no cambian.

Satélites: top-M bins por z de co-ocurrencia (C - E)/sqrt(E+0.5) vs
independencia (métrica elegida por diagnóstico: d de Cohen mediana 0.65
con M=100, frente a 0.52 diferencia y 0.39 lift en los primeros 256 bits).
"""
import numpy as np
from scipy import sparse
from scipy.special import ndtr

NBITS = 2048


def cargar_baseline(inter, nombre="topo_baseline_z_picos.npz"):
    d = np.load(inter / nombre)
    norm = str(d["normalizacion"]) if "normalizacion" in d else "nada"
    return {
        "normalizacion": norm,
        "sat_bins": d["sat_bins"],
        "sat_lift": d["sat_lift"],
        "mu_pos": d["mu_pos"].astype(np.float32),
        "sig_pos": d["sig_pos"].astype(np.float32),
        "mu_neg": d["mu_neg"].astype(np.float32),
        "sig_neg": d["sig_neg"].astype(np.float32),
        "cohens_d": d["cohens_d"],
        "n_b": d["n_b"],
        "n_sat": d["n_sat"],
        "max_lift": d["max_lift"],
        "level1": d["level1"],
        "M": int(d["M"][0]),
    }


def construir_S_ind(baseline, n_bins):
    """Indicador disperso (n_bins x 2048): 1 si el bin es satélite del bit."""
    sat = baseline["sat_bins"]
    n_bits, M = sat.shape
    bin_idx = sat.ravel()
    bit_idx = np.repeat(np.arange(n_bits, dtype=np.int32), M)
    ok = bin_idx >= 0
    return sparse.csr_matrix(
        (np.ones(int(ok.sum()), dtype=np.float32),
         (bin_idx[ok].astype(np.int32), bit_idx[ok])),
        shape=(n_bins, n_bits))


def agregar_max_por_mol(P, mol_codes):
    """Máximo de las filas de P por código de molécula (presencia si ALGÚN
    espectro muestra el bit). Devuelve (P_mol, códigos, conteos)."""
    codes = np.asarray(mol_codes)
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    uniq, starts = np.unique(cs, return_index=True)
    bounds = list(starts) + [len(cs)]
    counts = np.diff(bounds).astype(np.int32)
    Ps = P[order]
    Pm = np.empty((len(uniq), P.shape[1]), dtype=np.float32)
    s = 0
    for i, c in enumerate(counts):
        Pm[i] = Ps[s:s + c].max(axis=0)
        s += int(c)
    return Pm, uniq.astype(np.int32), counts


def predecir_huellas(Q_full, bins_sel, baseline, chunk=8192, scoring="binomial"):
    """Predice la huella por espectro.

    Q_full: CSR (n_spec x 200_000) con intensidades (valores >0 = presente).
    bins_sel: índices de los 50,000 bins seleccionados (mismo orden que el
      baseline). scoring: "binomial" (v2, defecto) usa sigma_efectiva =
      máx(sigma_empírica, sqrt(p0(1-p0)/n)) — test binomial de proporciones
      con piso por sobredispersión; "gaussiano" (v1) usa solo sigma_empírica.
    Devuelve (P (n_spec x 2048) float32, diag dict).
    """
    S = construir_S_ind(baseline, len(bins_sel))
    mu_neg, sig_neg = baseline["mu_neg"], baseline["sig_neg"]
    mu_pos, sig_pos = baseline["mu_pos"], baseline["sig_pos"]
    level1 = baseline["level1"]
    n_sat = baseline["n_sat"].astype(np.float32)
    normalizacion = baseline.get("normalizacion", "nada")
    n = Q_full.shape[0]
    P = np.empty((n, NBITS), dtype=np.float32)
    diag = {
        "n_present_L2": np.empty(n, dtype=np.int32),
        "n_present_L1": np.empty(n, dtype=np.int32),
        "n_capture": np.empty(n, dtype=np.int32),
        "n_picos": np.diff(Q_full.indptr).astype(np.int32),
        "n_picos_sel": np.empty(n, dtype=np.int32),
    }
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        Qc = (Q_full[s:e][:, bins_sel] > 0).astype(np.float32).tocsr()
        diag["n_picos_sel"][s:e] = np.diff(Qc.indptr).astype(np.int32)
        R = np.asarray((Qc @ S).todense(), dtype=np.float32)
        del Qc
        if normalizacion == "picos":
            R /= np.maximum(diag["n_picos_sel"][s:e][:, None].astype(np.float32), 1.0)
            n_bin = np.maximum(diag["n_picos_sel"][s:e].astype(np.float32)[:, None], 1.0)
        else:
            R /= n_sat[None, :]
            n_bin = np.maximum(n_sat[None, :], 1.0)
        if scoring == "binomial":
            sig_neg_eff = np.maximum(
                sig_neg[None, :],
                np.sqrt(np.maximum(mu_neg[None, :] * (1.0 - mu_neg[None, :]), 0.0) / n_bin))
            sig_pos_eff = np.maximum(
                sig_pos[None, :],
                np.sqrt(np.maximum(mu_pos[None, :] * (1.0 - mu_pos[None, :]), 0.0) / n_bin))
        else:  # gaussiano (v1)
            sig_neg_eff = sig_neg[None, :]
            sig_pos_eff = sig_pos[None, :]
        Zn = (R - mu_neg[None, :]) / sig_neg_eff
        present = Zn > 2.5
        diag["n_present_L2"][s:e] = present.sum(axis=1)
        diag["n_present_L1"][s:e] = (present & level1[None, :]).sum(axis=1)
        del present
        np.subtract(Zn, 2.5, out=Zn)
        ndtr(Zn, out=Zn)                      # score = Phi(Zn - 2.5)
        Zp = (R - mu_pos[None, :]) / sig_pos_eff
        del R
        diag["n_capture"][s:e] = (Zp < -2.5).sum(axis=1)
        del Zp
        P[s:e] = Zn
        del Zn
    return P, diag
