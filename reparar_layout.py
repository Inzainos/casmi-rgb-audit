"""Reparar layout plano en la Kali (CASMI sandbox).

Causa: un rsync con slash final (intermedios/ verificacion/ salida/)
aplano el contenido de los tres directorios en pruebas/. Este script
reubica cada archivo local en su ruta correcta segun el manifiesto
(md5, tamano, ruta relativa), comparando md5 cuando hay duplicados.

- Si el destino ya existe y coincide (md5+tamano): se borra el duplicado
  plano (solo si tambien coincide; si difiere se borra igual porque el
  destino ya es correcto, y se anota).
- Si el destino existe pero NO coincide y el plano SI coincide: se mueve
  el plano encima del destino.
- Si el destino no existe y el plano coincide: se mueve el plano.
- Si ninguno coincide o falta el archivo: se marca FLAG y NO se borra nada.
- Los archivos generados en la Kali (no estan en el manifiesto) no se tocan.

Uso: python3 reparar_layout.py --manifiesto local_manifiesto.tsv [--ejecutar]
Sin --ejecutar solo imprime el plan (dry-run).
"""
import argparse, hashlib, os, sys
from pathlib import Path

BASE = Path("/home/deamon/wt_deamonx/CASMI-sandbox/pruebas")


def md5sum(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def coincide(p: Path, md5: str, size: int) -> bool:
    try:
        if not p.is_file():
            return False
        if p.stat().st_size != size:
            return False
        return md5sum(p) == md5
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifiesto", required=True)
    ap.add_argument("--ejecutar", action="store_true",
                    help="aplicar los cambios (sin esto: dry-run)")
    args = ap.parse_args()
    stats = {"ok_en_sitio": 0, "mover": 0, "mover_encima": 0,
             "dup_borrado": 0, "dup_borrado_diferente": 0,
             "flag_falta": 0, "flag_mal": 0}
    flags = []
    for line in open(args.manifiesto):
        line = line.rstrip("\n")
        if not line:
            continue
        md5, size, rel = line.split("\t")
        size = int(size)
        rel = rel[2:] if rel.startswith("./") else rel
        dest = BASE / rel
        if "/" in rel:
            flat = BASE / rel.split("/", 1)[1]
        else:
            flat = dest
        if flat == dest:
            # archivo de raiz: solo verificar
            if coincide(dest, md5, size):
                stats["ok_en_sitio"] += 1
            else:
                stats["flag_mal"] += 1
                flags.append(f"MAL raiz: {rel}")
            continue
        dest_ok = coincide(dest, md5, size)
        flat_existe = flat.is_file()
        flat_ok = coincide(flat, md5, size) if flat_existe else False
        if dest_ok:
            stats["ok_en_sitio"] += 1
            if flat_existe:
                if flat_ok:
                    stats["dup_borrado"] += 1
                    print(f"DEL dup {flat} (== {rel})")
                    if args.ejecutar:
                        flat.unlink()
                else:
                    stats["dup_borrado_diferente"] += 1
                    print(f"DEL plano-diferente {flat} (destino {rel} OK)")
                    if args.ejecutar:
                        flat.unlink()
        elif flat_ok:
            if dest.exists():
                stats["mover_encima"] += 1
                print(f"MOVER-ENCIMA {flat} -> {dest} (destino no coincide)")
            else:
                stats["mover"] += 1
                print(f"MOVER {flat} -> {dest}")
            if args.ejecutar:
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(flat, dest)
        else:
            if not flat_existe and not dest.exists():
                stats["flag_falta"] += 1
                flags.append(f"FALTA: {rel} (ni destino ni plano)")
            else:
                stats["flag_mal"] += 1
                flags.append(f"MAL: {rel} (dest_ok={dest_ok} flat_ok={flat_ok} "
                             f"flat_existe={flat_existe})")
    print("=== RESUMEN ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"modo: {'EJECUTAR' if args.ejecutar else 'DRY-RUN'}")
    if flags:
        print("=== FLAGS ===")
        for f in flags:
            print(f)


if __name__ == "__main__":
    main()
