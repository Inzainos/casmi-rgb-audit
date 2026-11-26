"""CASMI paso 2 — 10_topo_baseline.py
Construye la RED DE REFERENCIA del modelo topológico (adaptación del
SNT Genomic Topologic Analyzer, repo Inzainos/Genomic_agent, MIT).

Mapeo genómico -> CASMI:
  paciente            -> molécula (su espectro)
  hub (gen regulador) -> la molécula misma
  satélite (gen)      -> bit de huella de Morgan (subestructura)
  TPM(sat)/TPM(hub)   -> R_b = fracción de los picos satélite del bit b
                         presentes en el espectro (observable por espectro)
  baseline sano n=40  -> librería de train (1 espectro por molécula):
                         para cada bit b, μ+/σ+ de R_b sobre moléculas CON b
                         y μ-/σ- sobre moléculas SIN b
  Z = (R - μ)/σ, |Z| > 2.5 (mismo umbral que el agente genómico)

  R_b (observable): --normalizar picos (defecto, fiel al cociente genómico
  R = TPM(sat)/TPM(hub), con el hub = la señal total del espectro):
    R_b = |S_b ∩ Q| / |Q ∩ bins_sel|  (fracción de los picos del espectro
    que son satélites del bit b; el denominador es la "TPM del hub").
  --normalizar nada (primera versión, descartada por diagnóstico):
    R_b = |S_b ∩ Q| / M  (sin denominador de hub; la riqueza del espectro
    infla R_b de todos los bits: corr(n_picos, n_presentes)=0.52).

Satélites de un bit: los top-M bins por lift L(p|b) = P(p|b)/P(p|¬b),
con pisos de soporte (co-ocurrencia >= 10, n_b >= 50). El lift baja los
bins ubicuos (fragmentos comunes) y sube los específicos del bit.

Nivel 1 (triage O(K)): bits bien caracterizados (n_b >= 200, d de Cohen
>= 0.5, lift máx >= 2.0) — el "oráculo" de asociaciones fuertes.
Nivel 2 (escáner por bloques): TODOS los 2048 bits, en bloques de 256
(el análogo de los bloques por cromosoma: acota la RAM por bloque).

Salida:
- intermedios/topo_baseline.npz: sat_bins (2048 x M int32, -1 = vacío),
  sat_lift (2048 x M float32), mu_pos/sig_pos/mu_neg/sig_neg (2048,),
  cohens_d (2048,), n_b (2048,), n_sat (2048,), level1 (2048, bool).
- verificacion/topo_baseline.json: estadísticas y diagnóstico.

Uso diagnóstico: --muestra N procesa solo los primeros N bits.
"""
import argparse, time, resource, json
from pathlib import Path
import numpy as np
from scipy import sparse

T0 = time.time()
def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time()-T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)

BASE = Path("/home/hatch/workspace/snt-taxonomia/kaggle-casmi")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"
NBITS = 2048
BLOQUE = 256

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=20, help="satélites por bit")
    ap.add_argument("--metrica", choices=["lift", "diferencia", "z"],
                    default="lift",
                    help="ranking de satélites: lift P(p|b)/P(p|¬b) | diferencia "
                         "P(p|b)-P(p|¬b) | z de co-ocurrencia vs independencia")
    ap.add_argument("--muestra", type=int, default=0, help="0 = todos los bits")
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--min-nb", type=int, default=50)
    ap.add_argument("--normalizar", choices=["picos", "nada"], default="picos",
                    help="denominador de R_b: picos = |Q| (hub); nada = M")
    args = ap.parse_args()
    M = args.m

    X = sparse.load_npz(INTER / "X_train_paso2.npz").tocsr()
    Y = sparse.load_npz(INTER / "y_train_paso2.npz").tocsr()
    n = X.shape[0]
    log(f"X_train: {X.shape[0]:,} x {X.shape[1]:,}, {X.nnz:,} nnz; "
        f"Y: {Y.shape[0]:,} x {Y.shape[1]:,}, {Y.nnz:,} nnz")
    Xb = (X > 0).tocsr()
    del X
    # OJO scipy: el producto disperso bool @ bool es OR-lógico (0/1), NO
    # conteo. Para la co-ocurrencia (conteos) se necesita float32.
    Xf = Xb.astype(np.float32)
    del Xb
    df_p = np.asarray(Xf.sum(axis=0)).ravel().astype(np.float64)
    n_b = np.asarray(Y.sum(axis=0)).ravel().astype(np.float64)
    Yt = Y.T.tocsr()
    Ytf = Yt.astype(np.float32)
    del Y
    log(f"df bins: media {df_p.mean():.1f}, máx {df_p.max():.0f}; "
        f"n_b: media {n_b.mean():.0f}, mín {n_b.min():.0f}, máx {n_b.max():.0f}, "
        f"bits con n_b<{args.min_nb}: {(n_b < args.min_nb).sum()}")

    n_bits = NBITS if args.muestra == 0 else min(args.muestra, NBITS)
    row_nnz = np.diff(Xf.indptr).astype(np.float32)  # picos seleccionados por espectro
    sat_bins = np.full((n_bits, M), -1, dtype=np.int32)
    sat_lift = np.zeros((n_bits, M), dtype=np.float32)
    mu_pos = np.zeros(n_bits, dtype=np.float32)
    sig_pos = np.zeros(n_bits, dtype=np.float32)
    mu_neg = np.zeros(n_bits, dtype=np.float32)
    sig_neg = np.zeros(n_bits, dtype=np.float32)
    cohens_d = np.zeros(n_bits, dtype=np.float32)
    n_sat = np.zeros(n_bits, dtype=np.int32)
    max_lift = np.zeros(n_bits, dtype=np.float32)

    # indicador disperso de satélites por bloque (para R): (n_bins x n_bits_bloque)
    for s in range(0, n_bits, BLOQUE):
        e = min(n_bits, s + BLOQUE)
        nb = e - s
        # co-ocurrencia: (nb x n_bins) = Ytf[s:e] @ Xf (conteos, float32)
        C = (Ytf[s:e] @ Xf).toarray().astype(np.float32)
        nb_arr = n_b[s:e][:, None].astype(np.float32)
        p_pos = C / np.maximum(nb_arr, 1.0)
        p_neg = (df_p[None, :].astype(np.float32) - C) / np.maximum(n - nb_arr, 1.0)
        p_neg = np.maximum(p_neg, np.float32(0.5 / max(n, 1)))
        lift = p_pos / p_neg
        if args.metrica == "lift":
            score = lift
        elif args.metrica == "diferencia":
            score = p_pos - p_neg
        else:  # z de co-ocurrencia vs independencia
            esp = nb_arr * df_p[None, :].astype(np.float32) / float(n)
            score = (C - esp) / np.sqrt(esp + 0.5)
        # pisos de soporte
        ok = (C >= args.min_count) & (nb_arr >= args.min_nb)
        score_ok = np.where(ok, score, -1.0)
        # top-M por lift
        for i in range(nb):
            row = score_ok[i]
            k = min(M, int((row > 0).sum()))
            if k == 0:
                continue
            top = np.argpartition(-row, k - 1)[:k]
            top = top[np.argsort(-row[top])]
            sat_bins[s + i, :k] = top.astype(np.int32)
            sat_lift[s + i, :k] = lift[i, top].astype(np.float32)
            n_sat[s + i] = k
            max_lift[s + i] = float(lift[i, top[0]]) if k else 0.0
        del C, p_pos, p_neg, lift, score, score_ok
        # R_b por espectro de entrenamiento para este bloque
        S_ind = np.zeros((Xf.shape[1], nb), dtype=np.float32)
        for i in range(nb):
            cols = sat_bins[s + i][sat_bins[s + i] >= 0]
            S_ind[cols, i] = 1.0
        R = Xf @ S_ind            # (n x nb) float32 denso: nº de satélites presentes
        if args.normalizar == "picos":
            R /= np.maximum(row_nnz[:, None], 1.0)
        else:
            R /= np.maximum(n_sat[s:e][None, :], 1).astype(np.float32)
        # máscaras con/sin bit
        Yb = Yt[s:e].T.toarray().astype(bool)   # (n x nb)
        for i in range(nb):
            m_pos = Yb[:, i]
            m_neg = ~m_pos
            rp = R[m_pos, i] if m_pos.any() else np.zeros(0, dtype=np.float32)
            rn = R[m_neg, i]
            mu_pos[s + i] = float(rp.mean()) if len(rp) else 0.0
            sig_pos[s + i] = float(rp.std()) if len(rp) > 1 else 0.0
            mu_neg[s + i] = float(rn.mean())
            sig_neg[s + i] = float(rn.std()) if len(rn) > 1 else 0.0
            sp = np.sqrt((sig_pos[s + i] ** 2 + sig_neg[s + i] ** 2) / 2.0)
            cohens_d[s + i] = float((mu_pos[s + i] - mu_neg[s + i]) / sp) if sp > 0 else 0.0
        del R, Yb
        log(f"bloque bits {s}-{e}: R y baselines listos "
            f"(d mediana parcial {np.median(cohens_d[s:e]):.2f})")

    level1 = (n_b[:n_bits] >= 200) & (cohens_d >= 0.5) & (max_lift >= 2.0)
    if args.muestra:
        suf = f"_{args.metrica}_muestra"
    elif args.metrica == "lift":
        suf = ""
    else:
        suf = f"_{args.metrica}"
    if args.normalizar == "picos":
        suf += "_picos"
    np.savez(INTER / f"topo_baseline{suf}.npz",
             sat_bins=sat_bins, sat_lift=sat_lift,
             mu_pos=mu_pos, sig_pos=sig_pos, mu_neg=mu_neg, sig_neg=sig_neg,
             cohens_d=cohens_d, n_b=n_b[:n_bits].astype(np.int64),
             n_sat=n_sat, max_lift=max_lift, level1=level1, M=np.asarray([M]),
             normalizacion=np.asarray(args.normalizar))
    stats = {
        "metrica": args.metrica,
        "normalizacion": args.normalizar,
        "M": M, "n_bits": int(n_bits), "min_count": args.min_count, "min_nb": args.min_nb,
        "n_sat_media": float(n_sat.mean()), "n_sat_mediana": float(np.median(n_sat)),
        "n_sat_cero": int((n_sat == 0).sum()),
        "cohens_d_media": float(cohens_d.mean()), "cohens_d_mediana": float(np.median(cohens_d)),
        "cohens_d_p25": float(np.percentile(cohens_d, 25)), "cohens_d_p75": float(np.percentile(cohens_d, 75)),
        "cohens_d_ge_0.5": int((cohens_d >= 0.5).sum()),
        "cohens_d_ge_1.0": int((cohens_d >= 1.0).sum()),
        "cohens_d_ge_2.0": int((cohens_d >= 2.0).sum()),
        "mu_pos_media": float(mu_pos.mean()), "mu_neg_media": float(mu_neg.mean()),
        "max_lift_mediana": float(np.median(max_lift)),
        "level1_n": int(level1.sum()),
    }
    (VERIF / f"topo_baseline{suf}.json").write_text(json.dumps(stats, indent=2))
    log(f"FIN baseline{suf}: {stats}")

if __name__ == "__main__":
    main()
