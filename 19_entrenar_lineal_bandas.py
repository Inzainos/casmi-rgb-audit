"""CASMI paso 3 — 19_entrenar_lineal_bandas.py
WPC Idea 2 (prueba barata): lineal SOLO con las 22 features de bandas
de 18_bandas.py (20 conteos por banda m/z + total + fracción ocupada).

Si las bandas solas superan el nulo con filtro de masa (0.0981 mol-split),
la señal es real y vale la pena el combinado (bandas + espectro fino).
Si no, la idea muere aquí sin tocar el 14.

Receta igual que 06/14 (SGD propio, lr 0.5, lam 1e-4, mom 0.9, 6 épocas,
batch 1024, semilla 42), con L2 por fila sobre las bandas (igual que las
features espectrales). K=22: W es 22x2048 (180 KB), el entrenamiento
tarda segundos.

Uso:
  python3 19_entrenar_lineal_bandas.py --tag bandas
Salida:
- intermedios/W_lineal_bandas.npy, intermedios/P_val_bandas.npy +
  P_val_mol_bandas.npy
- verificacion/entrenar_lineal_bandas.json
Reglas: nada a Kaggle/GitHub; log en verificacion/.
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

def l2_filas(X):
    norms = np.sqrt((X.astype(np.float64) ** 2).sum(axis=1))
    norms[norms <= 0] = 1.0
    return (X / norms[:, None]).astype(np.float32)

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
    ap.add_argument("--tag", default="bandas")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    log(f"entrenar bandas {args.tag}: epochs={args.epochs} lr={args.lr} "
        f"lam={args.lam} mom={args.momentum}")
    log(f"entorno: python {sys.version.split()[0]}, numpy {np.__version__}")
    B = np.load(INTER / "B_train_paso2.npy").astype(np.float32)
    B = l2_filas(B)
    Y = sparse.load_npz(INTER / "y_train_paso2.npz").tocsr()
    n, K = B.shape
    log(f"B_train {B.shape}, Y {Y.shape}")
    rng = np.random.default_rng(args.seed)
    W = np.zeros((K, NBITS), dtype=np.float32)
    V = np.zeros((K, NBITS), dtype=np.float32)
    n_batches = int(np.ceil(n / args.batch))
    perdidas = []
    for ep in range(args.epochs):
        perm = rng.permutation(n)
        se, cnt = 0.0, 0
        for b in range(n_batches):
            idxb = perm[b * args.batch:(b + 1) * args.batch]
            bs = len(idxb)
            Xb = B[idxb]
            scores = Xb @ W
            Yb = Y[idxb].toarray().astype(np.float32)
            err = scores - Yb
            se += float((err.astype(np.float64) ** 2).sum())
            cnt += err.size
            grad = (Xb.T @ err).astype(np.float32) / bs
            V = args.momentum * V + grad + args.lam * W
            W -= args.lr * V
        mse = se / max(cnt, 1)
        perdidas.append(mse)
        log(f"época {ep+1}: MSE {mse:.5f} |W|_F {float(np.linalg.norm(W)):.1f}")
    np.save(INTER / f"W_lineal_{args.tag}.npy", W)
    log(f"W guardado: {W.shape}, {W.nbytes/1e3:.0f} KB")
    (VERIF / f"entrenar_lineal_{args.tag}.json").write_text(json.dumps({
        "tag": args.tag, "epochs": args.epochs, "batch": args.batch,
        "lr": args.lr, "lam": args.lam, "momentum": args.momentum,
        "seed": args.seed, "mse_por_epoca": perdidas,
        "python": sys.version.split()[0], "numpy": np.__version__}, indent=2))
    Bv = np.load(INTER / "B_val_qmol.npy").astype(np.float32)
    Bv = l2_filas(Bv)
    m = np.load(INTER / "val_qmol_meta.npz", allow_pickle=True)
    mol_q = m["mol"].astype(np.int32)
    P = np.asarray(Bv @ W, dtype=np.float32)
    Pm, uniq, counts = agregar_por_mol(P, mol_q)
    np.save(INTER / f"P_val_{args.tag}.npy", Pm)
    np.save(INTER / f"P_val_mol_{args.tag}.npy", uniq)
    log(f"P_val_{args.tag}: {Pm.shape[0]:,} moléculas x {Pm.shape[1]:,}, "
        f"pred media {Pm.mean():.4f} max {Pm.max():.3f}")
    log(f"FIN entrenar bandas {args.tag}")

if __name__ == "__main__":
    main()
