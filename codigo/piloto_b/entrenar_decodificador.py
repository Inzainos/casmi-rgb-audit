#!/usr/bin/env python3
"""CASMI piloto B — entrenar_decodificador.py: decodificador huella Morgan -> SMILES.

Familia B del protocolo PROTOCOLO_GENERATIVO_DE_NOVO.md (§9): etapa 2
(huella -> SMILES). DECISIÓN DE ARQUITECTURA DE ELÁN (2026-11-12 ~18:34):
Transformer pequeño desde cero (NO VAE) — converge más rápido y maneja
mejor la sintaxis de SMILES largas. Se entrena sobre las 252,077 huellas
de GEN-TRAIN, arrancando con el piloto de 50k.

Modelo: encoder-decoder Transformer (PyTorch).
  - Encoder de huella: la huella Morgan 2048 bits (vector binario) se
    proyecta a K=8 vectores de memoria (d_model) con embeddings
    posicionales aprendidos. El decoder (TransformerDecoder estándar)
    atiende a esa memoria por cross-attention.
  - Decoder: genera el SMILES token a token (tokenizador SMILES por
    regex, vocabulario construido del train), con teacher forcing y
    cross-entropy (label smoothing 0.1).
  - Entrenamiento con huellas CORRUPTAS (protocolo §9 familia B,
    obligatorio): con probabilidad p_corrupt, la huella de entrada se
    corrompe para imitar el error de la etapa 1 (el predictor
    espectro->huella, kNN Tanimoto ~0.31 medido). La corrupción
    conserva una fracción de los bits activos y añade bits aleatorios
    hasta alcanzar un Tanimoto objetivo (~0.31, calibrado y reportado
    en el log). La mitad de las muestras se entrena con huella exacta.

Modos:
  entrenar  Entrena el decodificador (piloto 50k por defecto, o completo
            con --n_train 252077) y evalúa en el holdout interno (2% de
            GEN-TRAIN, semilla 42, ~5,041 moléculas).
  predecir  Carga un checkpoint y genera listas beam-25 por molécula
            (molecule_id,smiles con ";" — formato del evaluador
            30_evaluar_generativo.py) para una lista de moléculas.

Datos que lee (rutas --datos, por defecto el repo local):
  verificacion/split_arrays.npz   (mols, n_per_mol, val_mols, spec_mols)
  intermedios/mapa_smiles_full.parquet (inchikey14, smiles_rep)
  intermedios/huellas_pool.npz    (CSR bool 275810 x 2048, Morgan radio 2)
  verificacion/piloto_b/gen_val_verdad.parquet (solo modo predecir GEN-VAL)

Salida (--salida): modelo.pt (state_dict + config + vocabulario),
metricas.json, predicciones_holdout.csv, predicciones.csv (modo
predecir), log por stdout con tiempo y RSS.

Se ejecuta en Google Colab con GPU (runtime GPU, torch + rdkit
instalados por el notebook). En esta caja (sin GPU ni torch) solo se
pueden probar las piezas sin torch: tokenizador, corrupción y carga de
datos (el import de torch es perezoso y el script lo exige solo para
entrenar/predecir).
"""
import argparse
import csv
import json
import math
import re
import resource
import time
from pathlib import Path

import numpy as np

T0 = time.time()
SEMILLA = 42
NBITS = 2048
RADIO_MORGAN = 2


def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.time() - T0:8.1f}s | RSSmax {rss:7.0f} MB] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Tokenizador SMILES (regex estándar; sin dependencias)
# ---------------------------------------------------------------------------
SMILES_TOKEN_REGEX = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
)
TOKEN_PAD = "<PAD>"
TOKEN_BOS = "<BOS>"
TOKEN_EOS = "<EOS>"
TOKEN_UNK = "<UNK>"
TOKENS_ESPECIALES = [TOKEN_PAD, TOKEN_BOS, TOKEN_EOS, TOKEN_UNK]


def tokenizar_smiles(smiles):
    """SMILES -> lista de tokens. Devuelve (tokens, resto_sin_tokenizar)."""
    tokens = SMILES_TOKEN_REGEX.findall(smiles)
    resto = SMILES_TOKEN_REGEX.sub("", smiles)
    return tokens, resto


def construir_vocabulario(lista_smiles):
    """Vocabulario ordenado a partir de los SMILES de entrenamiento."""
    vocab = set(TOKENS_ESPECIALES)
    n_resto = 0
    for smi in lista_smiles:
        toks, resto = tokenizar_smiles(smi)
        vocab.update(toks)
        if resto:
            n_resto += 1
    vocab = sorted(vocab)
    return vocab, n_resto


class Tokenizador:
    def __init__(self, vocab):
        self.vocab = list(vocab)
        self.tok_a_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_a_tok = {i: t for i, t in enumerate(self.vocab)}
        self.pad = self.tok_a_id[TOKEN_PAD]
        self.bos = self.tok_a_id[TOKEN_BOS]
        self.eos = self.tok_a_id[TOKEN_EOS]
        self.unk = self.tok_a_id[TOKEN_UNK]

    def codificar(self, smiles, max_len):
        toks, _ = tokenizar_smiles(smiles)
        ids = [self.bos] + [self.tok_a_id.get(t, self.unk) for t in toks] + [self.eos]
        return ids[:max_len]

    def decodificar(self, ids):
        toks = []
        for i in ids:
            t = self.id_a_tok.get(int(i), TOKEN_UNK)
            if t in (TOKEN_BOS, TOKEN_PAD):
                continue
            if t == TOKEN_EOS:
                break
            toks.append(t)
        return "".join(toks)


# ---------------------------------------------------------------------------
# Corrupción de huella (imita el error de la etapa 1)
# ---------------------------------------------------------------------------
def corromper_huella(bits_on, t_objetivo, rng, r_conservar=None):
    """bits_on: array de índices de bits activos. Devuelve un array de
    índices corrompidos cuyo Tanimoto con el original es ~t_objetivo.

    Conserva una fracción r de los bits activos y añade bits aleatorios
    (de los apagados) hasta que el Tanimoto sea el objetivo:
    T = (r*n) / (n + m)  ->  m = (r*n / t) - n.
    """
    n = len(bits_on)
    if n == 0:
        return bits_on.copy()
    r = float(rng.uniform(0.4, 0.6)) if r_conservar is None else float(r_conservar)
    k_keep = max(1, int(round(r * n)))
    kept = rng.choice(bits_on, size=k_keep, replace=False)
    m = int(round(k_keep / max(t_objetivo, 1e-6) - n))
    m = max(0, m)
    if m > 0:
        apagados = np.setdiff1d(np.arange(NBITS, dtype=np.int32), bits_on,
                                assume_unique=False)
        m = min(m, len(apagados))
        added = rng.choice(apagados, size=m, replace=False)
        out = np.concatenate([kept, added]).astype(np.int32)
    else:
        out = kept.astype(np.int32)
    return np.sort(out)


def tanimoto_bits(a, b):
    """Tanimoto entre dos conjuntos de bits activos (arrays de índices)."""
    sa, sb = set(a.tolist()), set(b.tolist())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Carga de datos (sin torch)
# ---------------------------------------------------------------------------
def cargar_datos(ruta_datos):
    """Devuelve (mol_ik14, mapa_smiles, csr_huellas, val_set, spec_set).

    mol_ik14: array (275810,) de InChIKey14 en orden de molécula.
    mapa_smiles: dict inchikey14 -> smiles_rep.
    csr_huellas: scipy CSR bool (275810, 2048) o None si no existe.
    """
    from scipy import sparse
    import pyarrow.parquet as pq

    ruta = Path(ruta_datos)
    split = np.load(ruta / "verificacion" / "split_arrays.npz", allow_pickle=True)
    mol_ik14 = np.array([str(x) for x in split["mols"]])
    val_set = set(str(x) for x in split["val_mols"])
    spec_set = set(str(x) for x in split["spec_mols"])
    log(f"splits: {len(mol_ik14):,} moléculas; val {len(val_set):,}; "
        f"spec {len(spec_set):,}")
    tabla = pq.read_table(ruta / "intermedios" / "mapa_smiles_full.parquet")
    mapa = dict(zip(tabla.column("inchikey14").to_pylist(),
                    tabla.column("smiles_rep").to_pylist()))
    log(f"mapa_smiles_full: {len(mapa):,} SMILES representativos")
    ruta_huellas = ruta / "intermedios" / "huellas_pool.npz"
    csr = sparse.load_npz(ruta_huellas) if ruta_huellas.exists() else None
    if csr is not None:
        log(f"huellas_pool: CSR {csr.shape[0]:,} x {csr.shape[1]:,}, "
            f"nnz {csr.nnz:,}")
    else:
        log("AVISO: intermedios/huellas_pool.npz no existe")
    return mol_ik14, mapa, csr, val_set, spec_set


def construir_conjuntos(mol_ik14, mapa, csr, val_set, spec_set, semilla=SEMILLA):
    """GEN-TRAIN (moléculas con SMILES y huella no vacía, sin val ni spec),
    holdout interno (2% de GEN-TRAIN, semilla 42) y resto de entrenamiento.

    Devuelve dict con índices (posiciones en mol_ik14) y los arrays de
    bits activos por molécula (lista de arrays int32).
    """
    rng = np.random.default_rng(semilla)
    n_bits_por_mol = np.diff(csr.indptr) if csr is not None else np.zeros(len(mol_ik14), dtype=np.int64)
    gen_train = []
    for i, ik in enumerate(mol_ik14):
        if ik in val_set or ik in spec_set:
            continue
        if ik not in mapa:
            continue
        if n_bits_por_mol[i] == 0:
            continue
        gen_train.append(i)
    gen_train = np.array(gen_train, dtype=np.int64)
    perm = rng.permutation(len(gen_train))
    n_hold = max(1, int(round(0.02 * len(gen_train))))
    hold_idx = np.sort(gen_train[perm[:n_hold]])
    train_idx = np.sort(gen_train[perm[n_hold:]])
    log(f"GEN-TRAIN: {len(gen_train):,} moléculas con SMILES y huella; "
        f"holdout interno {len(hold_idx):,} (2%); entrenamiento {len(train_idx):,}")
    return {"gen_train": gen_train, "holdout": hold_idx, "train": train_idx}


def bits_de_molecula(csr, i):
    """Bits activos de la molécula i (array int32)."""
    return csr.indices[csr.indptr[i]:csr.indptr[i + 1]].astype(np.int32)


def vector_denso(bits, nbits=NBITS):
    """Bits activos -> vector binario float32 (nbits,)."""
    v = np.zeros(nbits, dtype=np.float32)
    if len(bits):
        v[bits] = 1.0
    return v


def binarizar_predicciones_rgb(predicciones, top_k=100):
    """Convierte scores RGB a huellas binarias con la regla lineal_bin."""
    p = np.asarray(predicciones, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != NBITS:
        raise ValueError(f"predicciones debe tener forma (n, {NBITS}); recibido {p.shape}")
    if not 1 <= top_k <= NBITS:
        raise ValueError(f"top_k debe estar entre 1 y {NBITS}")
    if not np.isfinite(p).all():
        raise ValueError("predicciones RGB contiene NaN o infinito")
    activas = []
    for fila in p:
        fila = np.clip(fila, 0.0, 1.0)
        kth = top_k if top_k < NBITS else top_k - 1
        idx = np.argpartition(-fila, kth)[:top_k]
        activas.append(np.sort(idx.astype(np.int32)))
    return activas


def mapear_huellas_por_molecule_id(molecule_ids, huellas, ids_objetivo):
    """Alinea huellas con consultas y falla explícitamente ante IDs ambiguos."""
    ids = [str(x) for x in molecule_ids]
    if len(ids) != len(huellas):
        raise ValueError("número de molecule_id y huellas no coincide")
    indice = {}
    for i, mid in enumerate(ids):
        if mid in indice:
            raise ValueError(f"molecule_id duplicado en predicciones RGB: {mid}")
        indice[mid] = i
    faltantes = sorted({str(mid) for mid in ids_objetivo} - indice.keys())
    if faltantes:
        muestra = ", ".join(faltantes[:5])
        raise ValueError(
            f"faltan {len(faltantes)} molecule_id en predicciones RGB: {muestra}"
        )
    return {str(mid): huellas[indice[str(mid)]] for mid in ids_objetivo}


# ---------------------------------------------------------------------------
# Modelo (requiere torch; import perezoso para poder probar las piezas
# sin torch en esta caja)
# ---------------------------------------------------------------------------
def _torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as exc:
        raise RuntimeError(
            "torch no está instalado: este script entrena en Colab con GPU. "
            "Instala torch (pip install torch) o corre el notebook de Colab."
        ) from exc


def construir_modelo(vocab_size, d_model=256, nhead=8, num_capas=4,
                     dim_ff=1024, dropout=0.1, k_memoria=8, max_len=160):
    """Transformer encoder-decoder huella->SMILES (torch)."""
    torch, nn = _torch()

    class CodificadorHuella(nn.Module):
        """Huella 2048 bits -> K vectores de memoria (d_model)."""

        def __init__(self):
            super().__init__()
            self.proyeccion = nn.Linear(NBITS, k_memoria * d_model)
            self.pos = nn.Parameter(torch.zeros(1, k_memoria, d_model))
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            b = x.shape[0]
            h = self.proyeccion(x).view(b, k_memoria, d_model)
            return self.norm(h + self.pos)

    class Decodificador(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = d_model
            self.k_memoria = k_memoria
            self.max_len = max_len
            self.emb_tokens = nn.Embedding(vocab_size, d_model)
            self.emb_pos = nn.Embedding(max_len, d_model)
            self.cod_huella = CodificadorHuella()
            capa = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=dropout, batch_first=True)
            self.decoder = nn.TransformerDecoder(capa, num_layers=num_capas)
            self.salida = nn.Linear(d_model, vocab_size)

        def memoria(self, huella):
            return self.cod_huella(huella)

        def forward(self, huella, tgt_ids, mascara_causal):
            mem = self.memoria(huella)
            pos = torch.arange(tgt_ids.shape[1], device=tgt_ids.device)
            h = self.emb_tokens(tgt_ids) + self.emb_pos(pos)[None, :, :]
            h = self.decoder(h, mem, tgt_mask=mascara_causal)
            return self.salida(h)

    return Decodificador()


def mascara_causal(n, dispositivo):
    torch, nn = _torch()
    return nn.Transformer.generate_square_subsequent_mask(n, device=dispositivo)


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def entrenar(args, ruta_datos, ruta_salida):
    torch, nn = _torch()
    dispositivo = args.device
    if dispositivo == "auto":
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"dispositivo: {dispositivo} (cuda disponible: "
        f"{torch.cuda.is_available()})")
    if dispositivo == "cpu":
        log("AVISO: entrenando en CPU — el piloto está pensado para GPU "
            "en Colab; en CPU será muy lento.")
    torch.manual_seed(args.semilla)
    np.random.seed(args.semilla)

    mol_ik14, mapa, csr, val_set, spec_set = cargar_datos(ruta_datos)
    conj = construir_conjuntos(mol_ik14, mapa, csr, val_set, spec_set,
                               semilla=args.semilla)
    idx_train_total = conj["train"]
    idx_hold = conj["holdout"]
    rng = np.random.default_rng(args.semilla)
    if args.n_train < len(idx_train_total):
        idx_train = np.sort(rng.choice(idx_train_total, size=args.n_train,
                                       replace=False))
    else:
        idx_train = idx_train_total
    log(f"piloto: {len(idx_train):,} moléculas de entrenamiento "
        f"(de {len(idx_train_total):,} disponibles)")

    # Vocabulario y longitudes.
    smiles_train = [mapa[mol_ik14[i]] for i in idx_train]
    vocab, n_resto = construir_vocabulario(smiles_train)
    tok = Tokenizador(vocab)
    log(f"vocabulario: {len(vocab)} tokens; SMILES con resto sin tokenizar: "
        f"{n_resto}")
    longitudes = []
    for smi in smiles_train[:5000]:
        longitudes.append(len(tokenizar_smiles(smi)[0]) + 2)
    longitudes = np.array(longitudes)
    max_len = int(min(args.max_len, max(32, np.percentile(longitudes, 99.9))))
    log(f"longitud de tokens (muestra 5k): mediana {np.median(longitudes):.0f}, "
        f"p99.9 {np.percentile(longitudes, 99.9):.0f} -> max_len {max_len}")

    # Bits activos por molécula de entrenamiento (lista de arrays).
    bits_train = [bits_de_molecula(csr, i) for i in idx_train]
    bits_hold = [bits_de_molecula(csr, i) for i in idx_hold]
    smiles_hold = [mapa[mol_ik14[i]] for i in idx_hold]

    modelo = construir_modelo(len(vocab), d_model=args.d_model,
                              nhead=args.nhead, num_capas=args.capas,
                              dim_ff=args.dim_ff, dropout=args.dropout,
                              k_memoria=args.k_memoria, max_len=max_len)
    modelo.to(dispositivo)
    n_param = sum(p.numel() for p in modelo.parameters())
    log(f"modelo: Transformer {args.capas} capas, d_model {args.d_model}, "
        f"{n_param:,} parámetros")
    optim = torch.optim.AdamW(modelo.parameters(), lr=args.lr,
                              weight_decay=0.01)
    pasos_por_epoca = max(1, math.ceil(len(idx_train) / args.batch))
    total_pasos = pasos_por_epoca * args.epocas
    warmup = min(500, total_pasos // 10)

    def lr_paso(paso):
        if paso < warmup:
            return args.lr * (paso + 1) / warmup
        prog = (paso - warmup) / max(1, total_pasos - warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * prog))

    criterio = nn.CrossEntropyLoss(ignore_index=tok.pad, label_smoothing=0.1)
    historial = []
    paso = 0
    for epoca in range(args.epocas):
        modelo.train()
        perm = rng.permutation(len(idx_train))
        perdida_acum, n_lotes = 0.0, 0
        for ini in range(0, len(idx_train), args.batch):
            lote_idx = perm[ini:ini + args.batch]
            # Huellas (densas) + corrupción.
            huellas = np.stack([
                vector_denso(
                    corromper_huella(bits_train[j], args.t_corrupt, rng)
                    if rng.random() < args.p_corrupt else bits_train[j])
                for j in lote_idx])
            # SMILES objetivo (teacher forcing).
            seqs = [tok.codificar(smiles_train[j], max_len) for j in lote_idx]
            largo = max(len(s) for s in seqs)
            tgt = np.full((len(seqs), largo), tok.pad, dtype=np.int64)
            for r, s in enumerate(seqs):
                tgt[r, :len(s)] = s
            tgt_t = torch.tensor(tgt, device=dispositivo)
            huella_t = torch.tensor(huellas, device=dispositivo)
            entrada = tgt_t[:, :-1]
            objetivo = tgt_t[:, 1:]
            masc = mascara_causal(entrada.shape[1], dispositivo)
            for g in optim.param_groups:
                g["lr"] = lr_paso(paso)
            optim.zero_grad()
            logits = modelo(huella_t, entrada, masc)
            perdida = criterio(logits.reshape(-1, logits.shape[-1]),
                               objetivo.reshape(-1))
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optim.step()
            perdida_acum += float(perdida.item())
            n_lotes += 1
            paso += 1
        perdida_media = perdida_acum / max(1, n_lotes)
        # Evaluación rápida en holdout (greedy, exact-match) cada N épocas.
        if (epoca + 1) % args.eval_cada == 0 or epoca + 1 == args.epocas:
            em, val = evaluar_holdout(modelo, tok, bits_hold, smiles_hold,
                                      dispositivo, args, n_eval=args.n_eval,
                                      modo="greedy")
            log(f"época {epoca + 1}/{args.epocas}: pérdida {perdida_media:.4f} | "
                f"holdout greedy exact-match {em:.4f} validez {val:.4f}")
            historial.append({"epoca": epoca + 1, "perdida": perdida_media,
                              "holdout_exact_match_greedy": em,
                              "holdout_validez": val})
        else:
            log(f"época {epoca + 1}/{args.epocas}: pérdida {perdida_media:.4f}")
            historial.append({"epoca": epoca + 1, "perdida": perdida_media,
                              "holdout_exact_match_greedy": None,
                              "holdout_validez": None})
    # Evaluación final en holdout (greedy + beam).
    em_g, val_g = evaluar_holdout(modelo, tok, bits_hold, smiles_hold,
                                  dispositivo, args, n_eval=args.n_eval,
                                  modo="greedy")
    em_b, val_b, tan_b = evaluar_holdout(modelo, tok, bits_hold, smiles_hold,
                                         dispositivo, args, n_eval=args.n_eval,
                                         modo="beam")
    log(f"FINAL holdout (n={min(args.n_eval, len(idx_hold))}): greedy "
        f"exact-match {em_g:.4f} validez {val_g:.4f} | beam-{args.beam} "
        f"exact-match {em_b:.4f} validez {val_b:.4f} Tanimoto@1 {tan_b:.4f}")
    # Guardado.
    ruta_salida.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": modelo.state_dict(),
        "config": {"d_model": args.d_model, "nhead": args.nhead,
                   "num_capas": args.capas, "dim_ff": args.dim_ff,
                   "dropout": args.dropout, "k_memoria": args.k_memoria,
                   "max_len": max_len, "n_train": int(len(idx_train)),
                   "epocas": args.epocas, "p_corrupt": args.p_corrupt,
                   "t_corrupt": args.t_corrupt, "semilla": args.semilla},
        "vocab": vocab,
    }, ruta_salida / "modelo.pt")
    metricas = {
        "n_train": int(len(idx_train)),
        "n_holdout": int(len(idx_hold)),
        "epocas": args.epocas,
        "vocab_size": len(vocab),
        "max_len": max_len,
        "parametros": int(n_param),
        "historial": historial,
        "holdout_final": {
            "n_eval": int(min(args.n_eval, len(idx_hold))),
            "greedy_exact_match": em_g, "greedy_validez": val_g,
            "beam_exact_match": em_b, "beam_validez": val_b,
            "beam_tanimoto_top1": tan_b,
        },
    }
    (ruta_salida / "metricas.json").write_text(json.dumps(metricas, indent=2))
    log(f"modelo guardado en {ruta_salida / 'modelo.pt'}; métricas en "
        f"{ruta_salida / 'metricas.json'}")
    return metricas


def evaluar_holdout(modelo, tok, bits_hold, smiles_hold, dispositivo, args,
                    n_eval=2000, modo="greedy"):
    """Exact-match y validez en holdout. Devuelve (exact_match, validez)
    y, en modo beam, también el Tanimoto@1 medio (Morgan)."""
    torch, nn = _torch()
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIO_MORGAN,
                                                    fpSize=NBITS)
    n = min(n_eval, len(bits_hold))
    idx = list(range(n))
    exactos, validos = 0, 0
    tanimoto_top1 = []
    modelo.eval()
    with torch.no_grad():
        for ini in range(0, n, 64):
            lote = idx[ini:ini + 64]
            huellas = np.stack([vector_denso(bits_hold[j]) for j in lote])
            huella_t = torch.tensor(huellas, device=dispositivo)
            if modo == "greedy":
                generados = generar_greedy(modelo, tok, huella_t, dispositivo,
                                           modelo.max_len)
            else:
                listas_beam, _ = generar_beam(
                    modelo, tok, huella_t, dispositivo, modelo.max_len,
                    beam=args.beam)
                generados = [cands[0] if cands else "" for cands in listas_beam]
            for j, gen_smi in zip(lote, generados):
                verdad = smiles_hold[j]
                if gen_smi == verdad:
                    exactos += 1
                mol = Chem.MolFromSmiles(gen_smi) if gen_smi else None
                if mol is not None:
                    validos += 1
                    if modo == "beam":
                        mol_v = Chem.MolFromSmiles(verdad)
                        if mol_v is not None:
                            b_v = set(gen.GetFingerprint(mol_v).GetOnBits())
                            b_c = set(gen.GetFingerprint(mol).GetOnBits())
                            union = b_v | b_c
                            tanimoto_top1.append(
                                len(b_v & b_c) / len(union) if union else 0.0)
    em = exactos / max(1, n)
    val = validos / max(1, n)
    tan = float(np.mean(tanimoto_top1)) if tanimoto_top1 else None
    return (em, val, tan) if modo == "beam" else (em, val)


# ---------------------------------------------------------------------------
# Generación (greedy + beam search por lotes)
# ---------------------------------------------------------------------------
def generar_greedy(modelo, tok, huella_t, dispositivo, max_len):
    """Decodificación greedy por lotes. Devuelve lista de SMILES."""
    torch, nn = _torch()
    modelo.eval()
    b = huella_t.shape[0]
    seqs = torch.full((b, 1), tok.bos, dtype=torch.long, device=dispositivo)
    terminadas = [False] * b
    with torch.no_grad():
        for _ in range(max_len - 1):
            if all(terminadas):
                break
            masc = mascara_causal(seqs.shape[1], dispositivo)
            logits = modelo(huella_t, seqs, masc)
            nxt = torch.argmax(logits[:, -1, :], dim=-1)
            seqs = torch.cat([seqs, nxt[:, None]], dim=1)
            for i in range(b):
                if int(nxt[i]) == tok.eos:
                    terminadas[i] = True
    return [tok.decodificar(seqs[i].tolist()) for i in range(b)]


def generar_beam(modelo, tok, huella_t, dispositivo, max_len, beam=25):
    """Beam search por lotes (ancho `beam`). Devuelve (listas_smiles,
    listas_scores): por molécula, las `beam` mejores secuencias por
    log-probabilidad acumulada (sin penalización de longitud)."""
    torch, nn = _torch()
    modelo.eval()
    b = huella_t.shape[0]
    # Estado por molécula: lista de (score, [ids]) activos y terminados.
    activos = [[(0.0, [tok.bos])] for _ in range(b)]
    terminados = [[] for _ in range(b)]
    with torch.no_grad():
        for _ in range(max_len - 1):
            # Lote de todas las secuencias activas (misma longitud L).
            planas = []
            dueno = []
            for m in range(b):
                for j, (_, seq) in enumerate(activos[m]):
                    planas.append(seq)
                    dueno.append(m)
            if not planas:
                break
            lote = torch.tensor(planas, dtype=torch.long, device=dispositivo)
            masc = mascara_causal(lote.shape[1], dispositivo)
            # Huellas expandidas al lote (cada secuencia activa lleva la
            # huella de su molécula).
            huella_lote = huella_t[torch.tensor(dueno, device=dispositivo)]
            logits = modelo(huella_lote, lote, masc)
            logp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
            logp = logp.cpu()
            # Por molécula: top-beam sobre (beam_activo x vocab).
            nuevos_activos = [[] for _ in range(b)]
            for m in range(b):
                idxs = [k for k, d in enumerate(dueno) if d == m]
                if not idxs:
                    continue
                puntajes = torch.tensor(
                    [activos[m][j][0] for j in range(len(activos[m]))])
                cand = puntajes[:, None] + logp[idxs]  # (W, V)
                plano = cand.reshape(-1)
                k = min(beam, plano.numel())
                mejores, pos = torch.topk(plano, k=k)
                for s, p in zip(mejores.tolist(), pos.tolist()):
                    j_beam, tok_id = divmod(p, logp.shape[1])
                    seq = activos[m][j_beam][1] + [tok_id]
                    if tok_id == tok.eos:
                        terminados[m].append((s, seq))
                    else:
                        nuevos_activos[m].append((s, seq))
                # Si todos los beams terminaron, no quedan activos.
                if not nuevos_activos[m] and not terminados[m]:
                    nuevos_activos[m] = activos[m]
            activos = nuevos_activos
            if all(not a for a in activos):
                break
    listas_smiles, listas_scores = [], []
    for m in range(b):
        cands = terminados[m] + activos[m]
        cands.sort(key=lambda x: x[0], reverse=True)
        cands = cands[:beam]
        listas_smiles.append([tok.decodificar(seq) for _, seq in cands])
        listas_scores.append([s for s, _ in cands])
    return listas_smiles, listas_scores


# ---------------------------------------------------------------------------
# Modo predecir
# ---------------------------------------------------------------------------
def predecir(args, ruta_datos, ruta_salida):
    torch, nn = _torch()
    dispositivo = args.device
    if dispositivo == "auto":
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"dispositivo: {dispositivo}")
    ckpt = torch.load(args.checkpoint, map_location=dispositivo,
                      weights_only=False)
    config = ckpt["config"]
    tok = Tokenizador(ckpt["vocab"])
    modelo = construir_modelo(len(ckpt["vocab"]), d_model=config["d_model"],
                              nhead=config["nhead"],
                              num_capas=config["num_capas"],
                              dim_ff=config["dim_ff"],
                              dropout=config["dropout"],
                              k_memoria=config["k_memoria"],
                              max_len=config["max_len"])
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.to(dispositivo)
    modelo.eval()
    log(f"checkpoint cargado: {args.checkpoint} "
        f"(n_train {config['n_train']}, épocas {config['epocas']})")

    mol_ik14, mapa, csr, _, _ = cargar_datos(ruta_datos)
    indice = {ik: i for i, ik in enumerate(mol_ik14)}
    import pyarrow.parquet as pq
    tabla = pq.read_table(args.lista)
    cols = tabla.column_names
    col_id = "molecule_id" if "molecule_id" in cols else "inchikey14"
    col_ik = "inchikey14" if "inchikey14" in cols else col_id
    ids = tabla.column(col_id).to_pylist()
    iks = tabla.column(col_ik).to_pylist()
    log(f"lista a predecir: {len(ids):,} moléculas ({args.lista})")
    rng = np.random.default_rng(args.semilla)
    if 0 < args.muestra < len(ids):
        sel = np.sort(rng.choice(len(ids), size=args.muestra, replace=False))
        ids = [ids[i] for i in sel]
        iks = [iks[i] for i in sel]
        log(f"muestra aleatoria (semilla {args.semilla}): {len(ids):,} "
            f"moléculas de {len(tabla):,}")
    huellas_rgb = None
    if args.predicciones_rgb:
        with np.load(args.predicciones_rgb, allow_pickle=False) as datos_rgb:
            if "molecule_id" not in datos_rgb or "predicciones" not in datos_rgb:
                raise ValueError(
                    "el NPZ RGB debe contener 'molecule_id' y 'predicciones'"
                )
            huellas_rgb = binarizar_predicciones_rgb(datos_rgb["predicciones"])
            huellas_rgb = mapear_huellas_por_molecule_id(
                datos_rgb["molecule_id"], huellas_rgb, ids
            )
        log(f"huellas RGB lineal_bin alineadas: {len(huellas_rgb):,}")
    ruta_salida.mkdir(parents=True, exist_ok=True)
    n_sin_huella, n_sin_mapa = 0, 0
    with open(ruta_salida / "predicciones.csv", "w", encoding="utf-8",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["molecule_id", "smiles"])
        for ini in range(0, len(ids), 32):
            lote_ids = ids[ini:ini + 32]
            lote_iks = iks[ini:ini + 32]
            filas, huellas = [], []
            for mid, ik in zip(lote_ids, lote_iks):
                if huellas_rgb is not None:
                    filas.append(mid)
                    huellas.append(vector_denso(huellas_rgb[str(mid)]))
                    continue
                # Búsqueda por molecule_id (clave DIRECTA): la columna
                # inchikey14 de la lista es la TAUTOMERIZADA (R-EVAL-6) y
                # el índice del pool es por clave directa. Buscar por
                # inchikey14 perdería 151/13,748 GEN-VAL y asignaría la
                # huella de OTRA molécula a otras 83 (medido 2026-11-12
                # con gen_val_verdad.parquet: 13,748/13,748 por
                # molecule_id vs 13,597/13,748 por inchikey14).
                i = indice.get(str(mid))
                if i is None or csr is None:
                    n_sin_mapa += 1
                    continue
                bits = bits_de_molecula(csr, i)
                if len(bits) == 0:
                    n_sin_huella += 1
                    continue
                if args.huella == "corrupta":
                    bits = corromper_huella(bits, args.t_corrupt, rng)
                filas.append(mid)
                huellas.append(vector_denso(bits))
            if not filas:
                continue
            huella_t = torch.tensor(np.stack(huellas), device=dispositivo)
            listas, _ = generar_beam(modelo, tok, huella_t, dispositivo,
                                     config["max_len"], beam=args.beam)
            for mid, cands in zip(filas, listas):
                w.writerow([mid, ";".join(cands)])
    log(f"predicciones escritas en {ruta_salida / 'predicciones.csv'} "
        f"(sin huella: {n_sin_huella}, sin mapa: {n_sin_mapa})")
    (ruta_salida / "prediccion_info.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "lista": str(args.lista),
        "n_moleculas": len(ids),
        "n_sin_huella": n_sin_huella,
        "n_sin_mapa": n_sin_mapa,
        "beam": args.beam,
        "huella": "rgb_lineal_bin_top100" if huellas_rgb is not None else args.huella,
        "predicciones_rgb": args.predicciones_rgb or None,
    }, indent=2))
    log("FIN — predecir")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="CASMI piloto B: decodificador huella Morgan -> SMILES "
                    "(Transformer desde cero, Elán 2026-11-12).")
    ap.add_argument("--modo", choices=["entrenar", "predecir"],
                    default="entrenar")
    ap.add_argument("--datos", default=str(Path(__file__).resolve().parents[2] / "datos"))
    ap.add_argument("--salida", default="verificacion/piloto_b/salida")
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--capas", type=int, default=4)
    ap.add_argument("--dim_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--k_memoria", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--p_corrupt", type=float, default=0.5)
    ap.add_argument("--t_corrupt", type=float, default=0.31)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--eval_cada", type=int, default=5,
                    help="evalúa el holdout cada N épocas (y siempre en la última)")
    ap.add_argument("--beam", type=int, default=25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--lista", default="")
    ap.add_argument("--predicciones-rgb", default="",
                    help="NPZ con molecule_id y scores RGB por molécula "
                         "(n x 2048); aplica clip [0,1] + top-100")
    ap.add_argument("--muestra", type=int, default=0,
                    help="0 = todas; N>0 = submuestra aleatoria de N "
                         "moléculas (semilla 42)")
    ap.add_argument("--huella", choices=["exacta", "corrupta"],
                    default="exacta")
    args = ap.parse_args()
    if args.predicciones_rgb and args.huella != "exacta":
        ap.error("--predicciones-rgb no se combina con --huella corrupta")
    ruta_datos = Path(args.datos)
    ruta_salida = Path(args.salida)
    log(f"piloto B decodificador huella->SMILES | modo {args.modo} | "
        f"datos {ruta_datos} | salida {ruta_salida}")
    if args.modo == "entrenar":
        entrenar(args, ruta_datos, ruta_salida)
    else:
        assert args.checkpoint and args.lista, \
            "predecir exige --checkpoint y --lista"
        predecir(args, ruta_datos, ruta_salida)
    log("FIN — entrenar_decodificador.py")


if __name__ == "__main__":
    main()
