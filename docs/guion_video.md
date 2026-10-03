# Guion de defensa (máximo 10 minutos)

**0:00–1:30 · Problema.** Mostrar una fotografía de aislador. Explicar que el prototipo asiste a quien revisa imágenes de líneas: clasifica un aislador delimitado manualmente como posible rotura y/o contaminación/flashover. No afirmar detección automática de objetos ni despliegue en ANDE. Las dos etiquetas pueden coexistir. La métrica de selección es macro-F1 en validación; en test mostrar también recall de cada defecto, porque una omisión importa para la priorización.

**1:30–3:00 · Datos y prevención de fuga.** Roboflow v2, licencia CC BY 4.0, COCO. Explicar que se encontraron variantes `h`, `v`, `d` de una misma escena en particiones originales. Se reagruparon 400 escenas y se dividieron 280/60/60; el aumento solo queda en entrenamiento. `GG` carece de cajas; `insulator` define el recorte, mientras `broken` y `pollution-flashover` son las dos salidas.

**3:00–5:30 · Código en GitHub y notebook.** Recorrer `src/data.py`, `src/features.py`, `src/model.py`, `src/train.py`, `src/finalize.py`, `app/streamlit_app.py`. Mostrar notebook ejecutado: carga de 100 filas, distribución, transformaciones de imágenes, carga de `model.pkl` y evaluación sobre la muestra test. Explicar que para imágenes se usa `torchvision.transforms`; `ColumnTransformer` y `Pipeline` de scikit-learn se emplean para datos tabulares y no se aplican directamente a los píxeles. Mostrar métricas completas desde `artifacts/metrics.json` y distinguirlas de las 20 observaciones test incluidas para reproducir la demo.

**5:30–8:30 · Demo.** Ejecutar `streamlit run app/streamlit_app.py`. Cargar una fotografía real del dataset, delimitar **un solo aislador** y mostrar probabilidades. Cargar otro ejemplo con un defecto distinto. Explicar que umbral 0,5 determina la alerta y requiere revisión humana.

**8:30–9:30 · ROI y próximos pasos.** Mostrar `docs/roi.md`: escenarios conservador, central y optimista basados en supuestos, sin atribuir ahorros a ANDE. Próximos pasos: 1) fotos paraguayas etiquetadas por especialistas; 2) medición del recall y falsos negativos en un piloto; 3) agregar localización automática para eliminar el recorte manual.

**9:30–10:00 · Cierre.** Recordar que el prototipo clasifica recortes y que la validación local y el ROI real siguen pendientes.
