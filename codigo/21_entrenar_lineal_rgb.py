"""CASMI ronda 2 — 21_entrenar_lineal_rgb.py
Lineal espectro -> huella Morgan con entrada RGB (3 canales por bin m/z).
Candidato (a) de wpc-schumann-ideas-cap37.md.

Variante de 14_entrenar_lineal_binario.py (mismo SGD propio con momento,
misma actualización in-place), con X = X_train_rgb.npz (261,536 x 3K) y
predicción con la misma transformación de 3 canales sobre val_qmol.npz.

Uso:
  python3 20_rgb.py --k-cols 20000
  python3 21_entrenar_lineal_rgb.py --tag rgb20k --epochs 6 --momentum 0.9
  python3 15_evaluar_pred.py --pred intermedios/P_val_rgb20k.npy \
      --mol intermedios/P_val_mol_rgb20k.npy --tag rgb20k --split mol [--bin]
Salida:
- intermedios/W_lineal_<tag>.npy
- intermedios/P_val_<tag>.npy + P_val_mol_<tag>.npy (13,748 x 2048)
- verificacion/entrenar_lineal_<tag>.json (config + MSE por época + versiones)
Reglas: nada a Kaggle/GitHub; log en verificacion/; un solo proceso pesado.
"""
import argparse, time, resource, json, sys
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

def l2_por_fila(X):
    """Igual que en 14: dividir por sqrt(suma de valores de la fila)."""
    norms = np.sqrt(np.asarray(X.sum(axis=1)).ravel().astype(np.float32))
    norms[norms <= 0] = 1.0
    X = sparse.diags((1.0 / norms).astype(np.float32)) @ X
    return X.tocsr().astype(np.float32)

def transformar_rgb(Q, bins_sel):
    """3 canales (sqrt / log1p / binario), cada uno normalizado por fila,
    concatenados [sqrt | log | bin]. Q es la librería bineada cruda."""
    Qb = Q[:, bins_sel].tocsr()
    canales = []
    for modo in ("sqrt", "log", "bin"):
        Xc = Qb.copy().tocsr()
        if modo == "sqrt":
            Xc.data = np.sqrt(Xc.data.astype(np.float32)).astype(np.float32)
        elif modo == "log":
            Xc.data = np.log1p(Xc.data.astype(np.float32)).astype(np.float32)
        else:
            Xc.data = np.ones_like(Xc.data, dtype=np.float32)
        canales.append(l2_por_fila(Xc))
    del Qb
    return sparse.hstack(canales, format="csr").astype(np.float32)

def agregar_por_mol(P, mol_codes):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rgb20k")
    ap.add_argument("--subset", type=int, default=0, help="0 = todas las filas")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log(f"entrenar RGB {args.tag}: epochs={args.epochs} batch={args.batch} "
        f"lr={args.lr} lam={args.lam} mom={args.momentum} seed={args.seed}")
    log(f"entorno: python {sys.version.split()[0]}, numpy {np.__version__}")

    X = sparse.load_npz(INTER / "X_train_rgb.npz").tocsr()
    Y = sparse.load_npz(INTER / "y_train_paso2.npz").tocsr()
    n_all = X.shape[0]
    n = args.subset if (args.subset and args.subset < n_all) else n_all
    if n < n_all:
        X, Y = X[:n].tocsr(), Y[:n].tocsr()
    K = X.shape[1]
    log(f"entrenar: n={n:,}, K={K:,} (3 canales), salidas={NBITS}")
    assert K % 3 == 0, "K debe ser múltiplo de 3 (3 canales)"
    assert Y.shape[0] == n_all, "Y no tiene las filas de 05"

    # --- SGD con momento, actualización in-place (como 14) ---
    rng = np.random.default_rng(args.seed)
    W = np.zeros((K, NBITS), dtype=np.float32)
    V = np.zeros((K, NBITS), dtype=np.float32) if args.momentum != 0.0 else None
    tmp = np.zeros((K, NBITS), dtype=np.float32) if args.momentum != 0.0 else None
    n_batches = int(np.ceil(n / args.batch))
    perdidas = []
    for ep in range(args.epochs):
        perm = rng.permutation(n)
        se = 0.0
        cnt = 0
        t_ep = time.time()
        for b in range(n_batches):
            idxb = perm[b * args.batch:(b + 1) * args.batch]
            bs = len(idxb)
            Xb = X[idxb]
            scores = Xb @ W
            Yb = Y[idxb].toarray().astype(np.float32)
            err = scores - Yb
            se += float((err.astype(np.float64) ** 2).sum())
            cnt += err.size
            grad = (Xb.T @ err).astype(np.float32)
            del Xb, scores, Yb, err
            if args.momentum != 0.0:
                np.multiply(V, args.momentum, out=V)
                np.divide(grad, bs, out=grad)
                V += grad
                np.multiply(W, args.lam, out=tmp)
                V += tmp
                np.multiply(V, args.lr, out=tmp)
                W -= tmp
            else:
                np.divide(grad, bs, out=grad)
                W -= args.lr * (grad + args.lam * W)
            del grad
            if (b + 1) % 64 == 0:
                log(f"época {ep+1}/{args.epochs} lote {b+1}/{n_batches} "
                    f"MSE parcial {se/max(cnt,1):.5f}")
        mse = se / max(cnt, 1)
        perdidas.append(mse)
        log(f"época {ep+1}: MSE {mse:.5f} ({time.time()-t_ep:.0f}s) |W|_F {float(np.linalg.norm(W)):.1f}")
    del V, tmp
    np.save(INTER / f"W_lineal_{args.tag}.npy", W)
    log(f"W guardado: {W.shape}, {W.nbytes/1e6:.0f} MB")

    config = {"tag": args.tag, "canales": ["sqrt", "log", "bin"],
              "subset": int(n), "n_train_total": int(n_all),
              "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
              "lam": args.lam, "momentum": args.momentum, "seed": args.seed,
              "mse_por_epoca": perdidas,
              "sin_momento_sin_V_tmp": bool(args.momentum == 0.0),
              "python": sys.version.split()[0], "numpy": np.__version__}
    (VERIF / f"entrenar_lineal_{args.tag}.json").write_text(json.dumps(config, indent=2))

    # --- predicción val mol-split (misma transformación RGB) ---
    bins_sel = np.load(INTER / "bins_rgb.npy")
    Q = sparse.load_npz(INTER / "val_qmol.npz").tocsr()
    m = np.load(INTER / "val_qmol_meta.npz", allow_pickle=True)
    mol_q = m["mol"].astype(np.int32)
    Qf = transformar_rgb(Q, bins_sel)
    del Q
    P = np.asarray(Qf @ W, dtype=np.float32)
    del Qf
    Pm, uniq, counts = agregar_por_mol(P, mol_q)
    del P
    np.save(INTER / f"P_val_{args.tag}.npy", Pm)
    np.save(INTER / f"P_val_mol_{args.tag}.npy", uniq)
    log(f"P_val_{args.tag}: {Pm.shape[0]:,} moléculas x {Pm.shape[1]:,}, "
        f"espectros/mol media {counts.mean():.2f}, "
        f"pred media {Pm.mean():.4f} max {Pm.max():.3f}")
    log(f"FIN entrenar RGB {args.tag}")

if __name__ == "__main__":
    main()
