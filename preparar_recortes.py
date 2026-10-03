from pathlib import Path
from PIL import Image

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

def etiquetar_aisladores(anotaciones, nombres, solapamiento):
    aisladores = [
        a for a in anotaciones
        if nombres[a["category_id"]] == "insulator"
    ]

    defectos = [
        a for a in anotaciones
        if nombres[a["category_id"]] in ("broken", "pollution-flashover")
    ]

    resultados = [
        {
            "bbox": aislador["bbox"],
            "broken": 0,
            "pollution-flashover": 0,
        }
        for aislador in aisladores
    ]

    for defecto in defectos:
        puntuaciones = [
            solapamiento(resultado["bbox"], defecto["bbox"])
            for resultado in resultados
        ]

        if puntuaciones and max(puntuaciones) > 0.5:
            mejor_indice = max(
                range(len(puntuaciones)),
                key=lambda i: puntuaciones[i],
            )

            nombre = nombres[defecto["category_id"]]
            resultados[mejor_indice][nombre] = 1
        else:
            nombre = nombres[defecto["category_id"]]
            resultados.append({
                "bbox": defecto["bbox"],
                "broken": int(nombre == "broken"),
                "pollution-flashover": int(
                    nombre == "pollution-flashover"
                ),
            })

    return resultados

def guardar_recortes(ruta_imagen, resultados, carpeta_destino, prefijo):
    carpeta_destino = Path(carpeta_destino)
    carpeta_destino.mkdir(parents=True, exist_ok=True)

    filas = []

    with Image.open(ruta_imagen) as original:
        imagen_limpia = original.convert("RGB")

    for indice, resultado in enumerate(resultados):
        x, y, ancho, alto = resultado["bbox"]
        margen = 15

        izquierda = max(0, int(x - margen))
        arriba = max(0, int(y - margen))
        derecha = min(imagen_limpia.width, int(x + ancho + margen))
        abajo = min(imagen_limpia.height, int(y + alto + margen))

        nombre_archivo = f"{prefijo}_{indice}.jpg"
        recorte = imagen_limpia.crop(
            (izquierda, arriba, derecha, abajo)
        )
        recorte.save(carpeta_destino / nombre_archivo)

        filas.append({
            "filename": nombre_archivo,
            "broken": resultado["broken"],
            "pollution-flashover": resultado["pollution-flashover"],
        })

    return filas