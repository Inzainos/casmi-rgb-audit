"""CASMI — 26_entrenar_lineal_ce.py
Experimento CE-como-feature (orden de Elán 2026-11-11 ~12:39).

Copia EXACTA del pipeline de 06_entrenar_lineal.py (mismo SGD con momento,
mismos hiperparámetros, mismo split semilla 42, mismo filtro de masa en la
evaluación posterior con 15_evaluar_pred.py), con UNA diferencia: añade
columna(s) CE a las features, después de la transformación sqrt+L2.

--ce-mode none   : referencia (sin CE; debe reproducir lineal_bin 0.4146 con --bin)
--ce-mode first  : 1 columna ce_first, clip[0,100]/100, sin CE -> 0.22
--ce-mode mean   : 1 columna ce_mean,  clip[0,100]/100, sin CE -> 0.40
--ce-mode onehot : 5 columnas one-hot {20, 40, 60, otro, sin CE} por ce_first
Las columnas CE de train/val las prepara 25_preparar_ce.py.
Solo predice val mol-split (la decisión es MRR@25 mol-split); NO predice spec.
Salida:
- intermedios/W_lineal_<tag>.npy ((K+n_ce) x 2048 float32)
- intermedios/P_val_<tag>.npy + P_val_mol_<tag>.npy (13,748 x 2048)
- verificacion/entrenar_lineal_<tag>.json (config + MSE por época + diseño CE)
Reglas: nada a Kaggle/GitHub; log en verificacion/; un solo proceso pesado.
"""
import argparse, time, resource, json
from pathlib import Path
import numpy as np
from scipy import sparse

T0 = time.time()
def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time()-T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)

BASE = Path("/home/deamon/wt_deamonx/CASMI-sandbox/pruebas")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"
NBITS = 2048
CE_MODES = ("none", "first", "mean", "onehot")

def transformar(Q, bins_sel):
    """sqrt + L2 por fila, solo columnas seleccionadas (idéntico a 06)."""
    Qf = Q[:, bins_sel].tocsr()
    Qf.data = np.sqrt(Qf.data.astype(np.float32)).astype(np.float32)
    norms = np.sqrt(np.asarray(Qf.sum(axis=1)).ravel().astype(np.float32))
    norms[norms <= 0] = 1.0
    Qf = sparse.diags((1.0 / norms).astype(np.float32)) @ Qf
    return Qf.tocsr().astype(np.float32)

def agregar_por_mol(P, mol_codes):
    """Media de las filas de P por código de molécula (idéntico a 06)."""
    codes = np.asarray(mol_codes)
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    uniq, starts = np.unique(cs, return_index=True)
    bounds = list(starts) + [len(cs)]
    counts = np.diff(bounds).astype(np.float32)
    rows = np.repeat(np.arange(len(uniq)), counts.astype(np.int64))
    cols = order.astype(np.int32)
    data = (1.0 / counts).astype(np.float32)
    data = np.repeat(data, counts.astype(np.int64))
    A = sparse.csr_matrix((data, (rows, cols)), shape=(len(uniq), len(codes)))
    Pm = A @ P
    return np.asarray(Pm, dtype=np.float32), uniq.astype(np.int32), counts.astype(np.int32)

def ce_sparse(mode, split):
    """Columna(s) CE como CSR float32. split: 'train' (261,536) o 'val' (38,377)."""
    if mode == "none":
        return None, 0
    if mode in ("first", "mean"):
        v = np.load(INTER / f"ce_{split}_{mode}.npy").astype(np.float32)
        return sparse.csr_matrix(v.reshape(-1, 1)), 1
    if mode == "onehot":
        oh = np.load(INTER / f"ce_{split}_onehot.npy").astype(np.float32)
        return sparse.csr_matrix(oh), 5
    raise ValueError(mode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ce-mode", choices=CE_MODES, default="none")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5.0)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=0, help="0 = todas las filas")
    args = ap.parse_args()

    X = sparse.load_npz(INTER / "X_train_paso2.npz").tocsr()
    Y = sparse.load_npz(INTER / "y_train_paso2.npz").tocsr()
    n = X.shape[0]
    if args.subset and args.subset < n:
        idx0 = np.arange(args.subset)
        X, Y = X[idx0], Y[idx0]
        n = args.subset
    K0 = X.shape[1]
    ce_tr, n_ce = ce_sparse(args.ce_mode, "train")
    if ce_tr is not None:
        if ce_tr.shape[0] != n:
            ce_tr = ce_tr[:n]
        X = sparse.hstack([X, ce_tr]).tocsr()
        log(f"CE train: modo={args.ce_mode}, columnas extra={n_ce}, " +
            f"media col CE={np.asarray(ce_tr.mean(axis=0)).ravel()}")
    K = X.shape[1]
    log(f"entrenar {args.tag}: n={n:,}, K={K:,} (base {K0:,} + CE {n_ce}), salidas={NBITS}, " +
        f"epochs={args.epochs}, batch={args.batch}, lr={args.lr}, lam={args.lam}, mom={args.momentum}")

    rng = np.random.default_rng(args.seed)
    W = np.zeros((K, NBITS), dtype=np.float32)
    V = np.zeros((K, NBITS), dtype=np.float32)
    n_batches = int(np.ceil(n / args.batch))
    perdidas = []
    for ep in range(args.epochs):
        perm = rng.permutation(n)
        se = 0.0
        cnt = 0
        t_ep = time.time()
        for b in range(n_batches):
            idx = perm[b * args.batch:(b + 1) * args.batch]
            bs = len(idx)
            Xb = X[idx]
            scores = Xb @ W
            Yb = Y[idx].toarray().astype(np.float32)
            err = scores - Yb
            se += float((err.astype(np.float64) ** 2).sum())
            cnt += err.size
            grad = (Xb.T @ err).astype(np.float32)
            del Xb, scores, Yb, err
            V = args.momentum * V + grad / bs + args.lam * W
            W -= args.lr * V
            del grad
            if (b + 1) % 64 == 0:
                log(f"época {ep+1}/{args.epochs} lote {b+1}/{n_batches} " +
                    f"MSE parcial {se/max(cnt,1):.5f}")
        mse = se / max(cnt, 1)
        perdidas.append(mse)
        log(f"época {ep+1}: MSE {mse:.5f} ({time.time()-t_ep:.0f}s) |W|_F {float(np.linalg.norm(W)):.1f}")
    del V
    np.save(INTER / f"W_lineal_{args.tag}.npy", W)
    log(f"W guardado: {W.shape}, {W.nbytes/1e6:.0f} MB")

    config = {"tag": args.tag, "ce_mode": args.ce_mode, "n_ce_cols": int(n_ce),
              "ce_diseno": ("clip[0,100]/100, sin CE->mediana (first 0.22, mean 0.40); " +
                            "onehot {20,40,60,otro,sin CE}" if args.ce_mode != "none" else None),
              "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
              "lam": args.lam, "momentum": args.momentum, "seed": args.seed,
              "subset": args.subset, "n_train": int(n), "K": int(K), "K_base": int(K0),
              "mse_por_epoca": perdidas}
    (VERIF / f"entrenar_lineal_{args.tag}.json").write_text(json.dumps(config, indent=2))

    # --- predicción val mol-split (misma transformación que 06) ---
    bins_sel = np.load(INTER / "bins_sel_paso2.npy")
    Q = sparse.load_npz(INTER / "val_qmol.npz").tocsr()
    m = np.load(INTER / "val_qmol_meta.npz", allow_pickle=True)
    mol_q = m["mol"].astype(np.int32)
    Qf = transformar(Q, bins_sel)
    del Q
    ce_v, _ = ce_sparse(args.ce_mode, "val")
    if ce_v is not None:
        Qf = sparse.hstack([Qf, ce_v]).tocsr()
    P = np.asarray(Qf @ W, dtype=np.float32)
    del Qf
    Pm, uniq, counts = agregar_por_mol(P, mol_q)
    del P
    np.save(INTER / f"P_val_{args.tag}.npy", Pm)
    np.save(INTER / f"P_val_mol_{args.tag}.npy", uniq)
    log(f"P_val_{args.tag}: {Pm.shape[0]:,} moléculas x {Pm.shape[1]:,}, " +
        f"espectros/mol media {counts.mean():.2f}, " +
        f"pred media {Pm.mean():.4f} max {Pm.max():.3f}")
    log(f"FIN entrenar lineal {args.tag} (ce-mode={args.ce_mode})")

if __name__ == "__main__":
    main()
