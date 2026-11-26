"""CASMI paso 3 — 14_entrenar_lineal_binario.py
Piloto WPC Idea 1: lineal espectro -> huella Morgan con espectro BINARIZADO.

Variante de 06_entrenar_lineal.py (mismo SGD propio, mismo loop), con:
- BASE corregida a ~/workspace/enveda-casmi-2026 (los 04-08 apuntan al
  path viejo snt-taxonomia/kaggle-casmi; este script NO toca ese path).
- --binario: features = (X > tau) sobre los valores sqrt+L2 ya guardados
  en X_train_paso2.npz, y luego L2 por fila. El umbral tau vive en la
  escala sqrt+L2 (misma transformación para entrenar y para predecir:
  primero sqrt+L2 como en 06, luego umbral, luego L2). Sin --binario es
  el control (sqrt+L2 tal cual).
- --k-cols: restringe a las top-K columnas por frecuencia de documento
  dentro de las 50k de 05 (K=20k por defecto: W+V = 328 MB, cabe con la
  RAM actual; el full K=50k necesita W+V = 820 MB y no cabe ahora).
- Actualización SGD IN-PLACE (np.multiply/divide con out=) para no crear
  temporales de 164 MB por operación: sin esto el pico de RAM del piloto
  rondaría 1.1 GB y mataría la caja.
- Predice solo val mol-split (la decisión del piloto es MRR@25 mol-split
  con filtro de masa, script 15).

Uso (piloto, un proceso pesado a la vez):
  python3 14_entrenar_lineal_binario.py --tag ctrl20k
  python3 14_entrenar_lineal_binario.py --tag bin005 --binario --tau 0.05
  python3 14_entrenar_lineal_binario.py --tag bin010 --binario --tau 0.10
Salida:
- intermedios/W_lineal_<tag>.npy, intermedios/bins_piloto_<tag>.npy
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

BASE = Path("/home/hatch/workspace/enveda-casmi-2026")
INTER = BASE / "intermedios"
VERIF = BASE / "verificacion"
NBITS = 2048

def l2_por_fila(X):
    """L2 por fila sobre CSR float32 (in-place en data)."""
    norms = np.sqrt(np.asarray(X.sum(axis=1)).ravel().astype(np.float32))
    norms[norms <= 0] = 1.0
    X = sparse.diags((1.0 / norms).astype(np.float32)) @ X
    return X.tocsr().astype(np.float32)

def transformar(Q, bins_sel, tau=None):
    """sqrt + L2 por fila (como 06); si tau no es None, binariza (X > tau)
    y re-normaliza L2. Q es la librería bineada cruda (intensidades)."""
    Qf = Q[:, bins_sel].tocsr()
    Qf.data = np.sqrt(Qf.data.astype(np.float32)).astype(np.float32)
    Qf = l2_por_fila(Qf)
    if tau is not None:
        Qf.data = (Qf.data > tau).astype(np.float32)
        Qf = l2_por_fila(Qf)
        Qf.eliminate_zeros()
    return Qf.tocsr().astype(np.float32)

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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--binario", action="store_true")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--k-cols", type=int, default=20_000)
    ap.add_argument("--subset", type=int, default=80_000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tau = args.tau if args.binario else None

    log(f"entrenar piloto {args.tag}: binario={args.binario} tau={tau} "
        f"K={args.k_cols:,} subset={args.subset:,} epochs={args.epochs} "
        f"lr={args.lr} lam={args.lam} mom={args.momentum}")
    log(f"entorno: python {sys.version.split()[0]}, numpy {np.__version__}, "
        f"scipy {sparse.__version__ if hasattr(sparse, '__version__') else 'n/a'}")

    # --- columnas: top k-cols por df dentro de las 50k de 05 ---
    bins_sel = np.load(INTER / "bins_sel_paso2.npy")
    X_full = sparse.load_npz(INTER / "X_train_paso2.npz").tocsr()
    df = np.asarray((X_full > 0).sum(axis=0)).ravel()
    idx = np.argsort(-df, kind="stable")[:args.k_cols]
    idx.sort()
    bins_piloto = bins_sel[idx].astype(np.int32)
    np.save(INTER / f"bins_piloto_{args.tag}.npy", bins_piloto)
    log(f"columnas: top {args.k_cols:,} por df (df mínimo incluido {int(df[idx].min())})")
    X = X_full[:, idx].tocsr()
    del X_full
    n_all = X.shape[0]
    n = args.subset if (args.subset and args.subset < n_all) else n_all
    if n < n_all:
        X = X[:n].tocsr()
    log(f"X: {X.shape[0]:,} x {X.shape[1]:,}, nnz {X.nnz:,}")

    # --- binarización (si aplica) sobre los valores sqrt+L2 guardados ---
    kept = None
    if args.binario:
        nnz_antes = X.nnz
        X.data = (X.data > tau).astype(np.float32)
        X.eliminate_zeros()
        kept = float(X.nnz / max(nnz_antes, 1))
        X = l2_por_fila(X)
        log(f"binarizado tau={tau}: nnz {nnz_antes:,} -> {X.nnz:,} "
            f"(queda {100*kept:.1f}% de los picos)")

    # --- objetivos (solo las filas del subset) ---
    Y_full = sparse.load_npz(INTER / "y_train_paso2.npz").tocsr()
    Y = Y_full[:n].tocsr() if n < Y_full.shape[0] else Y_full.tocsr()
    del Y_full
    log(f"Y: {Y.shape[0]:,} x {Y.shape[1]:,}, nnz {Y.nnz:,}")

    # --- SGD con momento, actualización in-place (sin temporales grandes) ---
    K = X.shape[1]
    rng = np.random.default_rng(args.seed)
    W = np.zeros((K, NBITS), dtype=np.float32)
    V = np.zeros((K, NBITS), dtype=np.float32)
    tmp = np.zeros((K, NBITS), dtype=np.float32)
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
            np.multiply(V, args.momentum, out=V)      # V *= mom
            np.divide(grad, bs, out=grad)             # grad /= bs
            V += grad
            np.multiply(W, args.lam, out=tmp)         # tmp = lam*W
            V += tmp
            np.multiply(V, args.lr, out=tmp)          # tmp = lr*V
            W -= tmp
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

    config = {"tag": args.tag, "binario": bool(args.binario), "tau": tau,
              "k_cols": int(args.k_cols), "subset": int(n), "n_train_total": int(n_all),
              "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
              "lam": args.lam, "momentum": args.momentum, "seed": args.seed,
              "fraccion_nnz_tras_umbral": kept,
              "mse_por_epoca": perdidas,
              "python": sys.version.split()[0], "numpy": np.__version__}
    (VERIF / f"entrenar_lineal_{args.tag}.json").write_text(json.dumps(config, indent=2))

    # --- predicción val mol-split (misma transformación) ---
    Q = sparse.load_npz(INTER / "val_qmol.npz").tocsr()
    m = np.load(INTER / "val_qmol_meta.npz", allow_pickle=True)
    mol_q = m["mol"].astype(np.int32)
    Qf = transformar(Q, bins_piloto, tau=tau)
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
    log(f"FIN entrenar piloto {args.tag}")

if __name__ == "__main__":
    main()
