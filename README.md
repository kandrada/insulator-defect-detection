# Detección de defectos en aisladores de líneas eléctricas

Proyecto final del Diplomado en Machine Learning y Deep Learning Aplicado,
Facultad de Ingeniería, UNA. 
Año 2026.

Alumna: Kimberly Andrada

## Problema

La revisión de fotografías de infraestructura eléctrica requiere identificar
aisladores con posibles defectos. Los aisladores dañados o contaminados pueden derivar en fallas de suministro eléctrico. La inspección actual es manual, lenta y poco sistemática.

Este proyecto desarrolla un prototipo de apoyo a esa revisión mediante clasificación de imágenes.

## Objetivo

El objetivo es clasificar recortes de aisladores según la presencia de dos tipos de defecto:
rotura (`broken`) y contaminación/flashover (`pollution-flashover`).

## Datos

Fuente: Insulator Defect, versión 2, disponible en Roboflow:

https://universe.roboflow.com/insulator-dataset/insulator-defect-yysit/dataset/2

Licencia del dataset: CC BY 4.0.

La descarga contiene imágenes y anotaciones en formato COCO. Las imágenes están en escala de grises. Las cajas de `insulator` se utilizan para generar recortes. Las cajas de defectos se relacionan con los recortes mediante un coeficiente de solapamiento superior a 0,5.

Si un defecto no puede asignarse a un aislador, su propia caja se utiliza para generar un recorte. Por ello, no todos los ejemplos representan necesariamente un aislador completo.

Las etiquetas son independientes: un recorte puede presentar ambos defectos.


## Modelo

Se utiliza MobileNetV3 Small con pesos preentrenados en ImageNet.

Reemplazamos la última capa por una capa con dos salidas y se comparan dos condiciones:

- Solo la última capa entrenable: cinco épocas.
- Todas las capas entrenables: tres épocas.

Se utiliza BCEWithLogitsLoss, AdamW con learning rate de 0,001 y lotes de 32 imágenes.

El criterio de selección es macro-F1 de validación, con umbral de 0,5 para cada etiqueta.

| Condición | Mejor época | Macro-F1 de validación |
|---|---:|---:|
| Solo última capa | 3 | 0,7652 |
| Ajuste de todas las capas | 3 | 0,6471 |

Se seleccionó el modelo que entrena únicamente la última capa. Tiene 1.519.906 parámetros totales, de los cuales 2.050 son entrenables durante esa condición.

## Resultados

El modelo seleccionado obtuvo macro-F1 de 0,7053 sobre 70 recortes de 60 grupos de escenas.

| Defecto | Precisión | Recall | F1 | Positivos |
|---|---:|---:|---:|---:|
| Rotura | 0,842 | 0,533 | 0,653 | 30 |
| Contaminación/flashover | 0,714 | 0,806 | 0,758 | 31 |

El modelo omitió 14 de los 30 recortes con rotura anotada y 6 de los 31 con contaminación/flashover.