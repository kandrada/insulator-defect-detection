# Decisiones verificadas

- Fuente: COCO `Insulator Defect v2`, licencia CC BY 4.0. Se rechaza la partición train/valid/test proporcionada por fuga entre vistas de la misma escena.
- `GG` aparece como categoría jerárquica sin anotaciones; `insulator` determina la caja del recorte. Predicción multietiqueta de dos defectos (`broken`, `pollution-flashover`).
- Una caja de defecto se asigna al aislador con mayor overlap coefficient (>0,5). Si no coincide con uno, su propia caja constituye un recorte. Se encontraron 11 de estas cajas huérfanas.
- Se detectaron 66 firmas dHash idénticas entre particiones originales y nombres relacionados con sufijos h/v/d. Se agrupan las 1600 fotografías originales declaradas en 400 identificadores de escena. Split nuevo 280/60/60 por escena, estratificado por combinación de etiquetas con semilla 42; train conserva los aumentos y valid/test usan un representante de cada escena. Resultan 3305/72/69 recortes. La auditoría dHash exacta en las 4162 imágenes arroja cero cruces entre nuevas particiones. No se descarta que haya otras fotografías muy parecidas entre escenas.
- Métrica de selección: macro-F1 de validación con umbral 0,5. Test solo después de seleccionar la condición y verificar tamaño del modelo.
- MobileNetV3 Small con pesos ImageNet compara `mobilenet_final` (solo capa final) y `mobilenet_full` (todas las capas), siguiendo las condiciones de M05·C8. El checkpoint íntegro cabe en 20 MB. ResNet18 (`final_only`, `full_finetune`) permanece como alternativa fuera del ZIP; `small_cnn` queda como baseline.
- El modelo clasifica un recorte manual y no detecta/localiza automáticamente defectos en toda una fotografía.
- No se realizó validación externa paraguaya; no se afirma generalización local. Grad-CAM queda como ampliación futura.
