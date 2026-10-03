import csv
from collections import Counter
from pathlib import Path

base = Path("data/processed")
filas = list(csv.DictReader((base / "labels.csv").open(encoding="utf-8")))

grupos = {"train": set(), "valid": set(), "test": set()}
faltantes = []
nombres_repetidos = set()
nombres_vistos = set()

for fila in filas:
    split = fila["split"]
    grupos[split].add(fila["original_group"])

    ruta = base / split / fila["filename"]
    if not ruta.is_file():
        faltantes.append(str(ruta))

    clave_archivo = (split, fila["filename"])
    if clave_archivo in nombres_vistos:
        nombres_repetidos.add(clave_archivo)
    nombres_vistos.add(clave_archivo)

print("Filas:", len(filas))
print("Archivos faltantes:", len(faltantes))
print("Nombres repetidos:", len(nombres_repetidos))

for split in ("train", "valid", "test"):
    parte = [fila for fila in filas if fila["split"] == split]
    combinaciones = Counter(
        (fila["broken"], fila["pollution-flashover"])
        for fila in parte
    )
    print(split, "recortes:", len(parte), "escenas:", len(grupos[split]))
    print("  combinaciones:", dict(combinaciones))

print("Cruces train/valid:", len(grupos["train"] & grupos["valid"]))
print("Cruces train/test:", len(grupos["train"] & grupos["test"]))
print("Cruces valid/test:", len(grupos["valid"] & grupos["test"]))