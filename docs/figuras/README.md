# Figuras del proyecto

Generadas con `scripts/checks/figuras.py` a partir de `results/` (no entrena nada).

## f01_muestra

![f01_muestra](f01_muestra.png)

Estrellas con vista válida por misión y clase (14,694 en total). NO_SIGNAL se fabricó 1:1 con los positivos.

## f02_galeria_local

![f02_galeria_local](f02_galeria_local.png)

Vista local de una estrella al azar por misión y clase (normalizada: la caída vale −1).

## f03_galeria_global

![f03_galeria_global](f03_galeria_global.png)

Vista global de las mismas estrellas: fase −0,5 a 0,5, tránsito en el centro.

## f04_atajos

![f04_atajos](f04_atajos.png)

Boosting sobre la vista global con el 20 % central tapado. Planeta ≈ 0,55–0,60 (sesgo de detección aceptado); FP de K2 0,84 es física (secundarios, variabilidad).

## f05_fuga_mision

![f05_fuga_mision](f05_fuga_mision.png)

Clasificador que intenta adivinar la misión con las vistas propias (IC 95 %). Motivó la ablación de la profundidad (#43) y la evaluación LOMO.

## f06_comparacion_modelos

![f06_comparacion_modelos](f06_comparacion_modelos.png)

Macro-F1 en val por misión (puntos) y media (raya). Seis familias se quedan en 0,78–0,85: el límite está en la información de la curva, no en el modelo.

## f07_prauc_por_clase

![f07_prauc_por_clase](f07_prauc_por_clase.png)

PR-AUC one-vs-rest por clase, media de las tres misiones, en val. Sin señal es fácil; planeta frente a FP es lo difícil.

## f08_confusion_logistica

![f08_confusion_logistica](f08_confusion_logistica.png)

Logística · vistas: filas = clase real, columnas = predicha (Pl. planeta, FP falso positivo, S/S sin señal); recuento y % por fila.

## f08_confusion_knn_pca

![f08_confusion_knn_pca](f08_confusion_knn_pca.png)

PCA + kNN · vistas: filas = clase real, columnas = predicha (Pl. planeta, FP falso positivo, S/S sin señal); recuento y % por fila.

## f08_confusion_boosting

![f08_confusion_boosting](f08_confusion_boosting.png)

Boosting · vistas: filas = clase real, columnas = predicha (Pl. planeta, FP falso positivo, S/S sin señal); recuento y % por fila.

## f08_confusion_mlp

![f08_confusion_mlp](f08_confusion_mlp.png)

MLP: filas = clase real, columnas = predicha (Pl. planeta, FP falso positivo, S/S sin señal); recuento y % por fila.

## f08_confusion_moment

![f08_confusion_moment](f08_confusion_moment.png)

MOMENT ajustado: filas = clase real, columnas = predicha (Pl. planeta, FP falso positivo, S/S sin señal); recuento y % por fila.

## f08_confusion_cnn

![f08_confusion_cnn](f08_confusion_cnn.png)

CNN por misión: filas = clase real, columnas = predicha (Pl. planeta, FP falso positivo, S/S sin señal); recuento y % por fila.

## f08_confusion_cnn_centroide

![f08_confusion_cnn_centroide](f08_confusion_cnn_centroide.png)

CNN + centroide: filas = clase real, columnas = predicha (Pl. planeta, FP falso positivo, S/S sin señal); recuento y % por fila.

## f09_centroide

![f09_centroide](f09_centroide.png)

CNN conjunta (semilla 0) con y sin el canal de centroide. Kepler (píxel de 4″) mejora; TESS (21″) y K2 (apuntado inestable) empeoran.

## f10_variantes_cnn

![f10_variantes_cnn](f10_variantes_cnn.png)

Variantes de la CNN conjunta (conjunto de 3 semillas cuando existe). Las diferencias están dentro del ruido de val.

## f11_errores_fp_kepler

![f11_errores_fp_kepler](f11_errores_fp_kepler.png)

FP de Kepler en val por motivo del Robovetter. Los de contaminación (co/ec, invisibles en la curva) son el techo físico del modelo solo curva.

## f12_lomo

![f12_lomo](f12_lomo.png)

LOMO: entrenar con dos misiones y evaluar en la tercera. La pérdida frente a ver la misión es pequeña en Kepler y K2; TESS es difícil incluso viéndola.

## f13_adaptacion_tess

![f13_adaptacion_tess](f13_adaptacion_tess.png)

Intentos de mejorar TESS sin sus etiquetas (fase 3). Barras de error: IC 95 % bootstrap cuando se midió.

## f14_curva_aprendizaje_tess

![f14_curva_aprendizaje_tess](f14_curva_aprendizaje_tess.png)

Kepler + K2 más una fracción del train de TESS. Pocas etiquetas apenas ayudan: pesan poco frente a ~10 000 de Kepler y K2.

## f15_test_final

![f15_test_final](f15_test_final.png)

CNN solo curva sin profundidad, conjunto de 5 semillas. IC 95 % bootstrap por estrella.

## f16_confusion_final_test

![f16_confusion_final_test](f16_confusion_final_test.png)

Modelo final en test. El fallo dominante es FP de TESS predicho como planeta (34 de 89).

## f17_bls

![f17_bls](f17_bls.png)

Con la efeméride de BLS, los NO_SIGNAL plegados en un pico de ruido parecen tránsitos: el modelo es un clasificador de señales ya detectadas (decisión 7).

## f18_negativos_dificiles

![f18_negativos_dificiles](f18_negativos_dificiles.png)

Añadir al entrenamiento NO_SIGNAL plegados en su pico de BLS. Decisión 13: base para TOIs, negativos difíciles solo para búsqueda.

## f19_errores_tipo_final

![f19_errores_tipo_final](f19_errores_tipo_final.png)

Kepler: motivos del Robovetter; TESS: comentarios de ExoFOP (etiquetas ruidosas). Las binarias de TESS se escapan como planeta un 31 %.

## f20_triaje

![f20_triaje](f20_triaje.png)

Val, todas las misiones. La combinación de la CNN sin plegar con SDE y la CNN plegada llega a 0,56 y toca techo (fase 5).

## f21_sensibilidad_snr

![f21_sensibilidad_snr](f21_sensibilidad_snr.png)

Tránsitos inyectados en curvas sin señal de val. Por encima de SNR 30 se detecta ~90 %.

## f22_multisector

![f22_multisector](f22_multisector.png)

IC 95 % bootstrap. Las estrellas sin sectores nuevos bajan lo mismo que las que ganan: es variación entre entrenamientos (auditoría A1).

## f23_transitos_individuales

![f23_transitos_individuales](f23_transitos_individuales.png)

Predicciones LOMO fuera de muestra. |z| par/impar, dispersión y tránsito más desviado (research/15). Ganancia en TESS +0,000.

## f24_tois_v1

![f24_tois_v1](f24_tois_v1.png)

Predicciones congeladas v1 (etiqueta v1 en GitHub). Con deriva > 0,25 la efeméride no sirve y aumentan NO_SIGNAL y FP (research/16).

## f25_esquema_lomo

![f25_esquema_lomo](f25_esquema_lomo.png)

Cada pliegue deja fuera una misión, elige el modelo solo con las otras dos y la evalúa una única vez.

## f26_etiquetas

![f26_etiquetas](f26_etiquetas.png)

Solo las etiquetas fuertes (confirmados y FP claros) se usan para evaluar; los candidatos son etiquetas débiles.

## f27_atajos_evolucion

![f27_atajos_evolucion](f27_atajos_evolucion.png)

Capacidad de distinguir los negativos fabricados sin mirar el tránsito, en cada versión de su construcción (research/05).

## f28_bls_periodo

![f28_bls_periodo](f28_bls_periodo.png)

Periodo de BLS frente al del catálogo en val. BLS acierta ~2 de cada 3; los FP dan más alias de ½ (binarias con dos eclipses parecidos).

## f29_calibracion_par_impar

![f29_calibracion_par_impar](f29_calibracion_par_impar.png)

Fracción de estrellas con diferencia par/impar significativa usando el ajuste tránsito a tránsito con ruido rojo (research/15).
