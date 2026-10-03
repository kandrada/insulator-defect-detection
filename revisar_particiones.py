import json
import re
from pathlib import Path

base = Path("data/raw/coco_v2")


def escena_original(nombre_archivo):
    # Quita el identificador añadido por Roboflow.
    nombre_base = nombre_archivo.split("_jpg.rf.")[0]

    # En este dataset, h/v/d aparecen en variantes de una escena.
    return re.sub(r"[hvd]$", "", nombre_base)


escenas = {}

for particion in ("train", "valid", "test"):
    archivo = base / particion / "_annotations.coco.json"

    with archivo.open(encoding="utf-8") as f:
        datos = json.load(f)

    escenas[particion] = {
        escena_original(imagen["file_name"])
        for imagen in datos["images"]
    }

    print(
        particion,
        "imágenes:", len(datos["images"]),
        "escenas:", len(escenas[particion]),
    )

print("Escenas repetidas entre conjuntos:")
print("train / valid:", len(escenas["train"] & escenas["valid"]))
print("train / test:", len(escenas["train"] & escenas["test"]))
print("valid / test:", len(escenas["valid"] & escenas["test"]))

import random

todas_las_escenas = sorted(
    escenas["train"] | escenas["valid"] | escenas["test"]
)

aleatorio = random.Random(42)
aleatorio.shuffle(todas_las_escenas)

train_nuevo = set(todas_las_escenas[:280])
valid_nuevo = set(todas_las_escenas[280:340])
test_nuevo = set(todas_las_escenas[340:])

print("\nNueva división por escena:")
print("train:", len(train_nuevo))
print("valid:", len(valid_nuevo))
print("test:", len(test_nuevo))

assert not (train_nuevo & valid_nuevo)
assert not (train_nuevo & test_nuevo)
assert not (valid_nuevo & test_nuevo)

print("No hay escenas compartidas.")

from collections import Counter

etiquetas_por_escena = {
    escena: set()
    for escena in todas_las_escenas
}

for particion in ("train", "valid", "test"):
    archivo = base / particion / "_annotations.coco.json"

    with archivo.open(encoding="utf-8") as f:
        datos = json.load(f)

    nombres = {
        categoria["id"]: categoria["name"]
        for categoria in datos["categories"]
    }

    escena_por_imagen = {
        imagen["id"]: escena_original(imagen["file_name"])
        for imagen in datos["images"]
    }

    for anotacion in datos["annotations"]:
        nombre = nombres[anotacion["category_id"]]

        if nombre in ("broken", "pollution-flashover"):
            escena = escena_por_imagen[anotacion["image_id"]]
            etiquetas_por_escena[escena].add(nombre)

combinaciones = Counter(
    tuple(sorted(etiquetas))
    for etiquetas in etiquetas_por_escena.values()
)

print("\nCombinaciones de defectos por escena:")
for combinacion, cantidad in combinaciones.items():
    print(combinacion or ("sin defectos anotados",), cantidad)

from sklearn.model_selection import train_test_split


def estrato(escena):
    etiquetas = etiquetas_por_escena[escena]
    return "+".join(sorted(etiquetas)) or "sin_defectos"


lista_escenas = sorted(todas_las_escenas)

train_estratificado, temporal = train_test_split(
    lista_escenas,
    test_size=0.30,
    random_state=42,
    stratify=[estrato(e) for e in lista_escenas],
)

valid_estratificado, test_estratificado = train_test_split(
    temporal,
    test_size=0.50,
    random_state=42,
    stratify=[estrato(e) for e in temporal],
)


def mostrar_distribucion(nombre, lista):
    conteo = Counter(estrato(e) for e in lista)
    print(nombre, len(lista), "escenas:", dict(conteo))


print("\nDivisión estratificada:")
mostrar_distribucion("train", train_estratificado)
mostrar_distribucion("valid", valid_estratificado)
mostrar_distribucion("test", test_estratificado)

assert not (set(train_estratificado) & set(valid_estratificado))
assert not (set(train_estratificado) & set(test_estratificado))
assert not (set(valid_estratificado) & set(test_estratificado))

from collections import defaultdict

asignacion = {}

for escena in train_estratificado:
    asignacion[escena] = "train"

for escena in valid_estratificado:
    asignacion[escena] = "valid"

for escena in test_estratificado:
    asignacion[escena] = "test"


imagenes_por_escena = defaultdict(list)

for carpeta_origen in ("train", "valid", "test"):
    archivo = base / carpeta_origen / "_annotations.coco.json"

    with archivo.open(encoding="utf-8") as f:
        datos = json.load(f)

    for imagen in datos["images"]:
        escena = escena_original(imagen["file_name"])
        imagenes_por_escena[escena].append((carpeta_origen, imagen))


imagenes_seleccionadas = {
    "train": [],
    "valid": [],
    "test": [],
}

for escena, opciones in imagenes_por_escena.items():
    destino = asignacion[escena]

    if destino == "train":
        imagenes_seleccionadas["train"].extend(opciones)
    else:
        elegida = min(
            opciones,
            key=lambda item: (
                item[0] == "train",
                item[0] != destino,
                item[1]["file_name"],
            ),
        )
        imagenes_seleccionadas[destino].append(elegida)

for destino, imagenes in imagenes_seleccionadas.items():
    print(destino, "archivos seleccionados:", len(imagenes))

anotaciones_por_imagen = defaultdict(list)
nombres_por_origen = {}

for carpeta_origen in ("train", "valid", "test"):
    archivo = base / carpeta_origen / "_annotations.coco.json"

    with archivo.open(encoding="utf-8") as f:
        datos = json.load(f)

    nombres_por_origen[carpeta_origen] = {
        categoria["id"]: categoria["name"]
        for categoria in datos["categories"]
    }

    for anotacion in datos["annotations"]:
        clave = (carpeta_origen, anotacion["image_id"])
        anotaciones_por_imagen[clave].append(anotacion)


for destino in ("train", "valid", "test"):
    carpeta_origen, imagen = imagenes_seleccionadas[destino][0]
    clave = (carpeta_origen, imagen["id"])
    anotaciones = anotaciones_por_imagen[clave]

    print(f"\nPrimera imagen del nuevo {destino}:")
    print("Carpeta física:", carpeta_origen)
    print("Archivo:", imagen["file_name"])
    print("Cantidad de cajas:", len(anotaciones))
    print(
        "Clases:",
        [
            nombres_por_origen[carpeta_origen][a["category_id"]]
            for a in anotaciones
        ],
    )

from preparar_recortes import etiquetar_aisladores, solapamiento

carpeta_origen, imagen = imagenes_seleccionadas["train"][0]
clave = (carpeta_origen, imagen["id"])

resultado = etiquetar_aisladores(
    anotaciones=anotaciones_por_imagen[clave],
    nombres=nombres_por_origen[carpeta_origen],
    solapamiento=solapamiento,
)

print("\nResultado de la función para la primera imagen:")
for aislador in resultado:
    print(aislador)

for destino, imagenes in imagenes_seleccionadas.items():
    total_recortes = 0

    for carpeta_origen, imagen in imagenes:
        clave = (carpeta_origen, imagen["id"])

        resultados = etiquetar_aisladores(
            anotaciones_por_imagen[clave],
            nombres_por_origen[carpeta_origen],
            solapamiento,
        )

        total_recortes += len(resultados)

    print(destino, "recortes previstos:", total_recortes)


from preparar_recortes import guardar_recortes

carpeta_origen, imagen = imagenes_seleccionadas["train"][0]
clave = (carpeta_origen, imagen["id"])

resultados = etiquetar_aisladores(
    anotaciones_por_imagen[clave],
    nombres_por_origen[carpeta_origen],
    solapamiento,
)

ruta_imagen = base / carpeta_origen / imagen["file_name"]

filas = guardar_recortes(
    ruta_imagen=ruta_imagen,
    resultados=resultados,
    carpeta_destino=Path("data/processed/prueba"),
    prefijo=f"{carpeta_origen}_{imagen['id']}",
)

print("\nFilas generadas:")
for fila in filas:
    print(fila)

import csv

carpeta_procesada = Path("data/processed")
todas_las_filas = []

for destino, imagenes in imagenes_seleccionadas.items():
    for numero, (carpeta_origen, imagen) in enumerate(imagenes, start=1):
        clave = (carpeta_origen, imagen["id"])

        resultados = etiquetar_aisladores(
            anotaciones_por_imagen[clave],
            nombres_por_origen[carpeta_origen],
            solapamiento,
        )

        ruta_imagen = base / carpeta_origen / imagen["file_name"]

        filas_imagen = guardar_recortes(
            ruta_imagen=ruta_imagen,
            resultados=resultados,
            carpeta_destino=carpeta_procesada / destino,
            prefijo=f"{carpeta_origen}_{imagen['id']}",
        )

        for fila in filas_imagen:
            fila["split"] = destino
            fila["source_file"] = imagen["file_name"]
            fila["original_group"] = escena_original(imagen["file_name"])

        todas_las_filas.extend(filas_imagen)

        if numero % 500 == 0:
            print(destino, "imágenes procesadas:", numero)


columnas = [
    "filename",
    "split",
    "source_file",
    "original_group",
    "broken",
    "pollution-flashover",
]

ruta_csv = carpeta_procesada / "labels.csv"

with ruta_csv.open("w", newline="", encoding="utf-8") as f:
    escritor = csv.DictWriter(f, fieldnames=columnas)
    escritor.writeheader()
    escritor.writerows(todas_las_filas)

print("\nCSV guardado en:", ruta_csv)
print("Filas totales:", len(todas_las_filas))
print("Recortes por conjunto:", Counter(fila["split"] for fila in todas_las_filas))