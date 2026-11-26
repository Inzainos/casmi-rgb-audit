"""CASMI paso 2 — 06_entrenar_lineal.py
Modelo lineal multi-salida espectro -> huella de Morgan, entrenado con
SGD por mini-lotes (implementación propia en float32).

¿Por qué implementación propia y no sklearn? SGDRegressor de sklearn solo
acepta y unidimensional (comprobado: ValueError con y (n, 2048)); hacer
2,048 ajustes separados sería 2,048 pasadas por los datos. Aquí un solo
modelo W (K x 2048) se actualiza con el gradiente de todas las salidas a la
vez: por lote, scores = Xb @ W; err = scores - Yb; grad = Xb.T @ err.

Pérdida: error cuadrático sobre bits 0/1 + L2 (lam). Optimizador: SGD con
momento. Features: sqrt(intensidad) + L2 por fila (las de 05).

Después del entrenamiento predice las huellas de las consultas de
validación (val_qmol y val_qspec): predicción por espectro y media por
molécula (para un modelo lineal es equivalente a predecir el espectro
medio, pero así vale para cualquier modelo).

Salida:
- intermedios/W_lineal_paso2.npy (K x 2048 float32)
- intermedios/P_val_lineal.npy + P_val_mol_lineal.npy (13,748 x 2048)
- intermedios/P_spec_lineal.npy + P_spec_mol_lineal.npy (9,936 x 2048)
- verificacion/entrenar_lineal.json (config + pérdida por época)
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

def transformar(Q, bins_sel):
    """sqrt + L2 por fila, solo columnas seleccionadas."""
    Qf = Q[:, bins_sel].tocsr()
    Qf.data = np.sqrt(Qf.data.astype(np.float32)).astype(np.float32)
    norms = np.sqrt(np.asarray(Qf.sum(axis=1)).ravel().astype(np.float32))
    norms[norms <= 0] = 1.0
    Qf = sparse.diags((1.0 / norms).astype(np.float32)) @ Qf
    return Qf.tocsr().astype(np.float32)

def agregar_por_mol(P, mol_codes):
    """Media de las filas de P por código de molécula. Devuelve (P_mol, códigos, conteos)."""
    codes = np.asarray(mol_codes)
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    uniq, starts = np.unique(cs, return_index=True)
    bounds = list(starts) + [len(cs)]
    counts = np.diff(bounds).astype(np.float32)
    # matriz de agregación dispersa (n_mol x n_filas) con 1/count
    rows = np.repeat(np.arange(len(uniq)), counts.astype(np.int64))
    cols = order.astype(np.int32)
    data = (1.0 / counts).astype(np.float32)
    data = np.repeat(data, counts.astype(np.int64))
    A = sparse.csr_matrix((data, (rows, cols)), shape=(len(uniq), len(codes)))
    Pm = A @ P
    return np.asarray(Pm, dtype=np.float32), uniq.astype(np.int32), counts.astype(np.int32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=0, help="0 = todas las filas")
    ap.add_argument("--sin-predecir", action="store_true")
    args = ap.parse_args()

    X = sparse.load_npz(INTER / "X_train_paso2.npz").tocsr()
    Y = sparse.load_npz(INTER / "y_train_paso2.npz").tocsr()
    n = X.shape[0]
    if args.subset and args.subset < n:
        idx0 = np.arange(args.subset)
        X, Y = X[idx0], Y[idx0]
        n = args.subset
    K = X.shape[1]
    log(f"entrenar: n={n:,}, K={K:,}, salidas={NBITS}, epochs={args.epochs}, "
        f"batch={args.batch}, lr={args.lr}, lam={args.lam}, mom={args.momentum}")

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
                log(f"época {ep+1}/{args.epochs} lote {b+1}/{n_batches} "
                    f"MSE parcial {se/max(cnt,1):.5f}")
        mse = se / max(cnt, 1)
        perdidas.append(mse)
        log(f"época {ep+1}: MSE {mse:.5f} ({time.time()-t_ep:.0f}s) |W|_F {float(np.linalg.norm(W)):.1f}")
    del V
    np.save(INTER / "W_lineal_paso2.npy", W)
    log(f"W guardado: {W.shape}, {W.nbytes/1e6:.0f} MB")

    config = {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
              "lam": args.lam, "momentum": args.momentum, "seed": args.seed,
              "subset": args.subset, "n_train": int(n), "K": int(K),
              "mse_por_epoca": perdidas}
    (VERIF / "entrenar_lineal.json").write_text(json.dumps(config, indent=2))

    if not args.sin_predecir:
        bins_sel = np.load(INTER / "bins_sel_paso2.npy")
        for tag, npz, meta in (("val", "val_qmol", "val_qmol_meta"),
                               ("spec", "val_qspec", "val_qspec_meta")):
            Q = sparse.load_npz(INTER / f"{npz}.npz").tocsr()
            m = np.load(INTER / f"{meta}.npz", allow_pickle=True)
            mol_q = m["mol"].astype(np.int32)
            Qf = transformar(Q, bins_sel)
            del Q
            P = np.asarray(Qf @ W, dtype=np.float32)
            del Qf
            Pm, uniq, counts = agregar_por_mol(P, mol_q)
            del P
            np.save(INTER / f"P_{tag}_lineal.npy", Pm)
            np.save(INTER / f"P_{tag}_mol_lineal.npy", uniq)
            log(f"P_{tag}_lineal: {Pm.shape[0]:,} moléculas x {Pm.shape[1]:,}, "
                f"espectros/mol media {counts.mean():.2f}, "
                f"pred media {Pm.mean():.4f} max {Pm.max():.3f}")
    log("FIN entrenar lineal")

if __name__ == "__main__":
    main()
