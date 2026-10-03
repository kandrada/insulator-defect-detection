# ROI exploratorio: revisión asistida de fotografías

**Escenario hipotético**, no medición de ANDE ni ahorro observado. El modelo actual requiere que una persona delimite cada aislador, por lo que podría aumentar el tiempo de revisión si la interfaz no se integra al proceso. Las cifras sirven para diseñar un piloto, no para justificar despliegue.

Supuestos ilustrativos: 10.000 fotografías/año; revisión manual de 2 minutos/foto; costo laboral completo de ₲40.000/h; costo anual del sistema y supervisión adicional ₲2.000.000. Se mantiene la captura de imágenes y se revisan manualmente todas las alertas; el ahorro neto de tiempo incluye recorte y revisión humana.

| Escenario | Reducción neta del tiempo por foto | Horas ahorradas/año | Valor del tiempo | Beneficio neto anual | ROI anual |
|---|---:|---:|---:|---:|---:|
| Conservador | 10% | 33,3 h | ₲1.333.333 | −₲666.667 | −33% |
| Central | 25% | 83,3 h | ₲3.333.333 | ₲1.333.333 | 67% |
| Optimista | 40% | 133,3 h | ₲5.333.333 | ₲3.333.333 | 167% |

Fórmulas: horas ahorradas = 10.000 × (2/60) × reducción; beneficio neto = horas ahorradas × ₲40.000 − ₲2.000.000; ROI = beneficio neto / ₲2.000.000. Redondeos a una decimal y guaraní entero. No se estima valor de interrupciones evitadas: faltan tasa de defectos, recall local y relación causal con cortes. Un piloto debe medir tiempo total con recorte manual, falsos negativos, costo de revisión y fotografías independientes de la red paraguaya antes de atribuir beneficios.
