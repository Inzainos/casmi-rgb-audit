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

def predecir_oof(holdout_codes, holdout_ids, W, bins_sel, ruta_salida,
                 batch_spectra=8192):
    """Predice el fold excluido usando todos sus espectros y promedia por molécula."""
    Q = sparse.load_npz(INTER / "val_lib.npz").tocsr()
    meta = np.load(INTER / "val_lib_meta.npz", allow_pickle=True)
    mol_codes = meta["mol"].astype(np.int32)
    rows = np.flatnonzero(np.isin(mol_codes, holdout_codes))
    if not len(rows):
        raise ValueError("el fold OOF no tiene espectros en val_lib")

    max_code = max(int(mol_codes.max()), int(holdout_codes.max()))
    pos_por_codigo = np.full(max_code + 1, -1, dtype=np.int32)
    pos_por_codigo[holdout_codes] = np.arange(len(holdout_codes), dtype=np.int32)
    destinos = pos_por_codigo[mol_codes[rows]]
    if (destinos < 0).any():
        raise ValueError("val_lib incluye códigos fuera del fold OOF")

    sumas = np.zeros((len(holdout_codes), W.shape[1]), dtype=np.float32)
    cuentas = np.zeros(len(holdout_codes), dtype=np.int32)
    log(f"predicción OOF: {len(rows):,} espectros para "
        f"{len(holdout_codes):,} moléculas")
    for ini in range(0, len(rows), batch_spectra):
        fin = min(ini + batch_spectra, len(rows))
        Qb = Q[rows[ini:fin]]
        Xb = transformar_rgb(Qb, bins_sel)
        Pb = np.asarray(Xb @ W, dtype=np.float32)
        destinos_b = destinos[ini:fin]
        np.add.at(sumas, destinos_b, Pb)
        np.add.at(cuentas, destinos_b, 1)
        if (ini // batch_spectra + 1) % 16 == 0 or fin == len(rows):
            log(f"predicción OOF: {fin:,}/{len(rows):,} espectros")
    if (cuentas == 0).any():
        raise ValueError(f"{int((cuentas == 0).sum())} moléculas OOF sin espectros")
    sumas /= cuentas[:, None]

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ruta_salida,
        molecule_id=np.asarray(holdout_ids, dtype=str),
        predicciones=sumas,
    )
    log(f"predicciones OOF guardadas: {ruta_salida} {sumas.shape}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rgb20k")
    ap.add_argument("--subset", type=int, default=0, help="0 = todas las filas")
    ap.add_argument("--holdout-ids", default="",
                    help="archivo con molecule_id a excluir y predecir OOF")
    ap.add_argument("--holdout-output", default="",
                    help="NPZ de salida con scores OOF por molecule_id")
    ap.add_argument("--batch-spectra-oof", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if bool(args.holdout_ids) != bool(args.holdout_output):
        ap.error("--holdout-ids y --holdout-output deben usarse juntos")
    if args.holdout_ids and args.subset:
        ap.error("--subset no se combina con el fold OOF")
    if args.batch_spectra_oof < 1:
        ap.error("--batch-spectra-oof debe ser al menos 1")

    log(f"entrenar RGB {args.tag}: epochs={args.epochs} batch={args.batch} "
        f"lr={args.lr} lam={args.lam} mom={args.momentum} seed={args.seed}")
    log(f"entorno: python {sys.version.split()[0]}, numpy {np.__version__}")

    X = sparse.load_npz(INTER / "X_train_rgb.npz").tocsr()
    Y = sparse.load_npz(INTER / "y_train_paso2.npz").tocsr()
    n_all = X.shape[0]
    holdout_codes = holdout_ids = None
    if args.holdout_ids:
        ids = [
            line.strip()
            for line in Path(args.holdout_ids).read_text().splitlines()
            if line.strip()
        ]
        if len(set(ids)) != len(ids):
            ap.error("--holdout-ids contiene molecule_id duplicados")
        train_mol = np.load(INTER / "train_mol_paso2.npy").astype(np.int32)
        mol_index = np.load(VERIF / "mol_index.npy", allow_pickle=True).astype(str)
        if len(train_mol) != n_all or Y.shape[0] != n_all:
            raise ValueError("X, Y y train_mol_paso2 no tienen filas alineadas")
        if (train_mol < 0).any() or (train_mol >= len(mol_index)).any():
            raise ValueError("train_mol_paso2 contiene códigos fuera de mol_index")
        train_ids = mol_index[train_mol]
        if len(set(train_ids)) != len(train_ids):
            raise ValueError("train_mol_paso2 repite molecule_id")
        holdout_mask = np.isin(train_ids, ids)
        missing = set(ids) - set(train_ids[holdout_mask])
        if missing:
            ap.error(f"{len(missing)} holdout molecule_id no aparecen en X_train")
        holdout_codes = train_mol[holdout_mask]
        holdout_ids = train_ids[holdout_mask]
        X = X[~holdout_mask].tocsr()
        Y = Y[~holdout_mask].tocsr()
        log(f"fold OOF: excluidas {len(holdout_ids):,} moléculas; "
            f"entrenamiento {X.shape[0]:,}")
    n = args.subset if (args.subset and args.subset < X.shape[0]) else X.shape[0]
    if n < X.shape[0]:
        X, Y = X[:n].tocsr(), Y[:n].tocsr()
    K = X.shape[1]
    log(f"entrenar: n={n:,}, K={K:,} (3 canales), salidas={NBITS}")
    assert K % 3 == 0, "K debe ser múltiplo de 3 (3 canales)"
    assert Y.shape[0] == n, "Y no quedó alineada con X tras filtrar el fold"

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
              "n_holdout_oof": int(len(holdout_ids)) if holdout_ids is not None else 0,
              "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
              "lam": args.lam, "momentum": args.momentum, "seed": args.seed,
              "mse_por_epoca": perdidas,
              "sin_momento_sin_V_tmp": bool(args.momentum == 0.0),
              "python": sys.version.split()[0], "numpy": np.__version__}
    (VERIF / f"entrenar_lineal_{args.tag}.json").write_text(json.dumps(config, indent=2))

    if holdout_codes is not None:
        del X, Y
        bins_sel = np.load(INTER / "bins_rgb.npy")
        predecir_oof(
            holdout_codes, holdout_ids, W, bins_sel, args.holdout_output,
            batch_spectra=args.batch_spectra_oof,
        )
        log(f"FIN entrenamiento OOF {args.tag}")
        return

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
