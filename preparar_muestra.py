from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parent
carpeta_procesada = ROOT / "data" / "processed"
carpeta_muestra = ROOT / "data" / "sample"

datos = pd.read_csv(
    carpeta_procesada / "labels.csv",
    dtype={"original_group": str},
)

# Conservamos todo el test para reproducir su evaluación.
test = datos.loc[datos["split"] == "test"].copy()
assert len(test) == 70, "Se esperaban 70 recortes de test."

# Elegimos ejemplos de los otros conjuntos sin usar predicciones.
train = datos.loc[datos["split"] == "train"].sample(
    n=15, random_state=42
)
valid = datos.loc[datos["split"] == "valid"].sample(
    n=15, random_state=42
)

muestra = pd.concat([train, valid, test], ignore_index=True)

assert len(muestra) == 100
assert not muestra.duplicated(["split", "filename"]).any()

# Comprobamos los archivos antes de comenzar a copiarlos.
for fila in muestra.itertuples(index=False):
    origen = carpeta_procesada / fila.split / fila.filename
    if not origen.is_file():
        raise FileNotFoundError(f"Falta el recorte: {origen}")

# Conservamos la estructura train/valid/test.
for fila in muestra.itertuples(index=False):
    origen = carpeta_procesada / fila.split / fila.filename
    destino = carpeta_muestra / fila.split / fila.filename

    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, destino)

# El CSV obligatorio queda en la raíz.
ruta_csv = ROOT / "datos_muestra.csv"
muestra.to_csv(ruta_csv, index=False)

print("CSV guardado:", ruta_csv)
print("Filas:", len(muestra))
print("Recortes seleccionados por conjunto:")
print(muestra["split"].value_counts())

peso_imagenes = sum(
    (carpeta_muestra / fila.split / fila.filename).stat().st_size
    for fila in muestra.itertuples(index=False)
)
print(f"Peso de las 100 imágenes: {peso_imagenes / 1_000_000:.2f} MB")