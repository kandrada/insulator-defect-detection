import json
from PIL import Image, ImageDraw
from collections import Counter
from pathlib import Path

archivo = Path("data/raw/coco_v2/train/_annotations.coco.json")

with archivo.open(encoding="utf-8") as f:
    datos = json.load(f)

print("Imágenes:", len(datos["images"]))
print("Anotaciones:", len(datos["annotations"]))
print("Categorías:", datos["categories"])

nombres = {
    categoria["id"]: categoria["name"]
    for categoria in datos["categories"]
}

conteo = Counter()

for anotacion in datos["annotations"]:
    id_categoria = anotacion["category_id"]
    nombre = nombres[id_categoria]
    conteo[nombre] += 1

print("Cajas anotadas por categoría:")
for nombre, cantidad in conteo.items():
    print(f"  {nombre}: {cantidad}")

imagen = datos["images"][0]
print("\nPrimera imagen:")
print("ID:", imagen["id"])
print("Archivo:", imagen["file_name"])
print("Tamaño:", imagen["width"], "x", imagen["height"])

print("\nCajas de esa imagen:")
for anotacion in datos["annotations"]:
    if anotacion["image_id"] == imagen["id"]:
        nombre = nombres[anotacion["category_id"]]
        print(nombre, anotacion["bbox"])

ruta_imagen = Path("data/raw/coco_v2/train") / imagen["file_name"]

with Image.open(ruta_imagen) as original:
    foto = original.convert("RGB")

dibujo = ImageDraw.Draw(foto)

for anotacion in datos["annotations"]:
    if anotacion["image_id"] != imagen["id"]:
        continue

    nombre = nombres[anotacion["category_id"]]
    x, y, ancho, alto = anotacion["bbox"]

    color = "red" if nombre == "broken" else "lime"
    dibujo.rectangle(
        [x, y, x + ancho, y + alto],
        outline=color,
        width=3,
    )
    dibujo.text((x, max(0, y - 15)), nombre, fill=color)

foto.save("primera_imagen_anotada.jpg")
print("\nImagen guardada: primera_imagen_anotada.jpg")

def solapamiento(caja_a, caja_b):
    ax, ay, aw, ah = caja_a
    bx, by, bw, bh = caja_b

    izquierda = max(ax, bx)
    arriba = max(ay, by)
    derecha = min(ax + aw, bx + bw)
    abajo = min(ay + ah, by + bh)

    ancho_interseccion = max(0, derecha - izquierda)
    alto_interseccion = max(0, abajo - arriba)
    area_interseccion = ancho_interseccion * alto_interseccion

    area_a = aw * ah
    area_b = bw * bh

    return area_interseccion / min(area_a, area_b)

anotaciones_imagen = [
    a for a in datos["annotations"]
    if a["image_id"] == imagen["id"]
]

aisladores = [
    a for a in anotaciones_imagen
    if nombres[a["category_id"]] == "insulator"
]

defectos = [
    a for a in anotaciones_imagen
    if nombres[a["category_id"]] in ("broken", "pollution-flashover")
]

etiquetas_por_aislador = [
    {"broken": 0, "pollution-flashover": 0}
    for _ in aisladores
]

for defecto in defectos:
    puntuaciones = [
        solapamiento(aislador["bbox"], defecto["bbox"])
        for aislador in aisladores
    ]

    mejor_indice = max(
        range(len(puntuaciones)),
        key=lambda i: puntuaciones[i],
    )

    if puntuaciones[mejor_indice] > 0.5:
        nombre_defecto = nombres[defecto["category_id"]]
        etiquetas_por_aislador[mejor_indice][nombre_defecto] = 1

for numero, etiquetas in enumerate(etiquetas_por_aislador, start=1):
    print(f"Aislador {numero}: {etiquetas}")

carpeta_recortes = Path("pruebas_recortes")
carpeta_recortes.mkdir(exist_ok=True)

with Image.open(ruta_imagen) as original:
    imagen_limpia = original.convert("RGB")

for numero, aislador in enumerate(aisladores, start=1):
    x, y, ancho, alto = aislador["bbox"]
    margen = 15

    izquierda = max(0, int(x - margen))
    arriba = max(0, int(y - margen))
    derecha = min(imagen_limpia.width, int(x + ancho + margen))
    abajo = min(imagen_limpia.height, int(y + alto + margen))

    recorte = imagen_limpia.crop(
        (izquierda, arriba, derecha, abajo)
    )

    destino = carpeta_recortes / f"aislador_{numero}.jpg"
    recorte.save(destino)

    print(destino, etiquetas_por_aislador[numero - 1])