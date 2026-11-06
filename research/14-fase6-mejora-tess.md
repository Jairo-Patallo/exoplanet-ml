# Fase 6: mejora del modelo en TESS

Punto de partida: macro-F1 de TESS 0,72 en test (oficial, `research/10`) y 0,784 en val (receta final entrenada con train, 5 semillas). Todo se mide en val; el test no se toca.

## 1. Varios sectores por estrella (opción A, decisión de Jairo 2026-11-06)

**Por qué la opción A.** Con todos los sectores, el número de sectores delata la clase. Mediana por clase: planetas 7, FP 3, sin señal 2. Solo con esa variable se separa planeta de FP con un ROC-AUC de 0,71. Por eso cada pareja (positivo y su NO_SIGNAL) usa el mismo número de sectores: el mínimo realmente disponible en MAST para las dos. Cada estrella usa su sector original más otros al azar (semilla fija).

**Montaje:**
- `scripts/data/build_tess_multisector.py` → `data/sample/views_tess_multi`, `results/sample/qc_tess_multi.csv`.
- Se usan 714 parejas de train, val y test, sin errores. Mediana de 2–3 sectores por estrella; la mediana de tránsitos pasa a 16–24.
- El resto de estrellas conserva su vista de 1 sector.
- Entrenamiento: misma receta, solo train, 5 semillas (`scripts/models/tess_multisector.py`, pesos en `data/models/multi_seed*.pt`).
- Comparación: `scripts/checks/tess_multi_compare.py` → `results/models/tess_multi_compare.csv` y `tess_multi_shortcut.csv`.

**Criterio fijado antes de mirar:** el macro-F1 de TESS en val sube con el IC de la diferencia sin 0, Kepler y K2 no empeoran, y el atajo queda controlado.

| Val | n | 1 sector | Multisector | Diferencia [IC 95 %] |
|---|---|---|---|---|
| TESS, todas | 302 | 0,784 | 0,765 | −0,019 [−0,059, +0,018] |
| TESS, ganan sectores | 134 | 0,823 | 0,810 | −0,014 [−0,093, +0,063] |
| TESS, sin sectores nuevos | 168 | 0,750 | 0,725 | −0,025 [−0,065, +0,008] |
| Kepler | 1 116 | 0,874 | 0,870 | — |
| K2 | 69 | 0,891 | 0,907 | — |

F1 de FALSE_POSITIVE en TESS: 0,684 → 0,641.

**Control del atajo** (ROC-AUC con una sola variable; 0,5 = no separa):

| Tarea | Número de sectores | Ruido, 1 sector | Ruido, multisector |
|---|---|---|---|
| Planeta vs FP | 0,573 | 0,586 | 0,602 |
| Candidato vs sin señal | 0,495 | 0,527 | 0,525 |

**Conclusión: el criterio no se cumple, así que no se adopta.**
- No hay mejora apreciable: la diferencia es negativa pero compatible con 0.
- Las estrellas que no ganan sectores bajan casi lo mismo que las que ganan (−0,025 frente a −0,014). La diferencia parece variación entre entrenamientos, no un efecto de los sectores.
- El atajo quedó controlado como se esperaba: candidato vs sin señal, ~0,5; planeta vs FP, residual de ~0,57–0,60.
- Diferencia con la fase 3 (+0,034 con 2 sectores): allí el modelo LOMO, sin TESS en el entrenamiento, se evaluaba con más tránsitos en las mismas estrellas. Aquí se reentrena también con vistas multisector, y solo 134 estrellas de val ganan sectores. Con ese tamaño, un efecto de +0,03 no se distingue del ruido (el IC tiene ±0,08 de ancho).
- Las vistas de 1 sector siguen siendo las del modelo. Las vistas multisector quedan guardadas.

## 2. Todas las etiquetas de TESS: aparcada (opción B, decisión de Jairo 2026-11-06)

En ExoFOP hay 885 TOIs etiquetados fuera del dataset. Con datos de 2 min solo hay ~412 KP (planetas conocidos, profundos, fáciles) y ~1 FP. Los ~280 FP nuevos solo tienen imágenes completas (QLP), y mezclarlas crearía un atajo por tipo de producto de datos. La ganancia esperada es pequeña o nula, así que la tarea queda en stand-by.

## 3. Calibración con temperature scaling por misión (decisión de Jairo 2026-11-06)

`scripts/models/temperature.py`:
- p_T = softmax(log p / T), con una T por misión y una global de respaldo para telescopios nuevos.
- Se ajusta con las predicciones de val de la receta entrenada solo con train (5 semillas).
- Se evalúa con 5 folds dentro de val, fuera de pliegue.
- No cambia la clase predicha: el macro-F1 y el test oficial no cambian.
- Salidas: `data/models/final_temperature.json`, `results/models/calibration_final.csv` y `.png`.

| Val | T | ECE antes → después | NLL antes → después | Confianza media / acierto (antes) |
|---|---|---|---|---|
| Kepler | 0,77 | 0,039 → **0,017** | 0,290 → 0,279 | 0,86 / 0,90 |
| K2 | 0,68 | 0,084 → 0,058 | 0,351 → 0,343 | 0,81 / 0,90 |
| TESS | 1,07 | 0,063 → 0,056 | 0,534 → 0,540 | 0,79 / 0,82 |
| Global (respaldo) | 0,80 | 0,040 → **0,020** | 0,342 → 0,335 | 0,84 / 0,88 |

**Lectura:**
- El ensemble de 5 semillas es **poco confiado** en Kepler y K2: dice 0,86 y acierta el 0,90. Promediar las probabilidades de varias semillas las suaviza. T < 1 las agudiza y deja el ECE de Kepler en 0,017.
- **TESS ya está casi calibrado** (T ≈ 1; la NLL fuera de pliegue no mejora). La sobreconfianza de TESS vista en la fase 3 era del modelo **LOMO**, que no había visto TESS. Este modelo sí lo ve al entrenar.
- Limitación: las T se ajustan con el ensemble entrenado solo con train y se aplican al modelo final (train + val), que podría tener una confianza algo distinta. Se revisará con la validación prospectiva (fase 7).

## Cierre de la fase 6

- Varios sectores: no mejora.
- Más etiquetas: aparcada.
- Calibración: hecha (Kepler y K2 mejoran; TESS ya estaba bien).

El modelo final sigue siendo el de la fase 4, ahora con las temperaturas por misión.
