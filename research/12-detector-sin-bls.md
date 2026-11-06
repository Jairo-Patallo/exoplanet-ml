# Detector sin BLS: CNN 1D sobre la curva sin plegar (fase 4)

**Objetivo (Jairo, 2026-11-06):** separar CANDIDATO (PLANET o FP) de SIN_SEÑAL solo con la curva, sin efeméride. Las falsas alarmas al final de la cadena deben quedar ≤ 1 % sin perder candidatos. La línea base, SDE de BLS + CNN plegada con negativos difíciles (`research/11`), recupera el 47 % de los candidatos con un 1 % de falsas alarmas. Su límite es BLS: en el 25 % de los candidatos no encuentra el periodo en ninguno de sus 3 mejores picos.

## Montaje

- **Curvas** (`scripts/data/build_series.py`): los mismos archivos que las vistas, aplanados a ciegas (sin efeméride), en bloques de 30 min.
  - Los archivos se encadenan con 2 d de hueco, hasta 9 216 bloques (192 d).
  - Total: 13 186 estrellas de train y val. Medianas: TESS 1 330 bloques, K2 3 800, Kepler 9 216 (2 trimestres).
- **Entrada:** flujo en unidades del ruido de la estrella (mediana y MAD), comprimido con signo·log1p, más un canal de máscara.
- **Red** (`scripts/models/series_cnn.py`): CNN 1D de 7 bloques (9 216 → 72), con max y media globales. Salida binaria.
- **Entrenamiento:**
  - estrellas reales de train;
  - en cada época, una copia de cada curva SIN_SEÑAL con un tránsito inyectado al vuelo (etiqueta CANDIDATO): periodo log-uniforme, duración física, forma trapezoidal, SNR total log-uniforme de 5 a 50 (comprobado: la SNR medida coincide con la pedida);
  - parada temprana con el 15 % de las estrellas de train;
  - 17 s por época; paró en la época 25 (mejor: 19). Una semilla.

## Resultados en val

Mismas 1 403 estrellas que la línea base. Las combinaciones son regresiones logísticas puntuadas fuera de pliegue (5 folds).

| Puntuación | ROC-AUC | Recall con 1 % de falsas alarmas | Con 5 % | Ruido que pasa con recall 90 / 95 / 99 % |
|---|---|---|---|---|
| SDE + CNN plegada (línea base) | 0,877 | 0,473 | 0,662 | 46 / 67 / 95 % |
| CNN sin plegar | 0,840 | **0,554** | 0,613 | 59 / 75 / 91 % |
| CNN sin plegar + SDE | 0,874 | 0,536 | 0,657 | 55 / 68 / 90 % |
| **Las tres** | **0,891** | **0,592** | **0,724** | 47 / 65 / 91 % |

Por misión, CNN sin plegar con un 1 % de falsas alarmas: Kepler 0,54, K2 0,43, TESS 0,57. En TESS, ROC-AUC 0,89.

**Sensibilidad** (tránsitos inyectados en las curvas SIN_SEÑAL de val, umbral del 1 % de falsas alarmas reales):

| SNR | 5–7 | 7–10 | 10–15 | 15–20 | 20–30 | 30–50 |
|---|---|---|---|---|---|---|
| Detectados | 17 % | 36 % | 54 % | 66 % | 81 % | **93 %** |

**Candidatos reales por SNR estimada** (las tres puntuaciones, umbral del 1 %): SNR > 50: **95 %** (248); 30–50: 78 %; 20–30: 62 %. Por debajo, la SNR estimada (profundidad de la vista local) es poco fiable.

## Conclusiones

- **El detector sin BLS funciona:** solo, ya supera a la línea base en el punto exigente (1 % de falsas alarmas). Combinado con BLS y la CNN plegada, sube de 0,47 a **0,59**. Mira cosas distintas que BLS. *(Corregido abajo: con 5 semillas, 0,525 sola y 0,564 combinada; la semilla 0 fue afortunada.)*
- **Señales claras:** con un 1 % de falsas alarmas detecta el 93 % de los tránsitos inyectados con SNR 30–50 y el 95 % de los candidatos reales con SNR > 50.
- **Lo que falta:** con recall alto (95 %) aún pasa el 65 % del ruido. Las señales débiles (SNR < 15) se detectan la mitad de las veces o menos.
- Siguientes pasos baratos: varias semillas (~8 min cada una), red más ancha y SNR de inyección más baja. Si no basta, Transformer sobre esta misma entrada.

## Variantes: semillas, inyecciones débiles y red ancha (2026-11-06)

`series_cnn.py --tag {base, weak, wide}` y `scripts/checks/series_ensemble.py` → `results/models/series_ensemble*.csv`.
- **base:** 5 semillas.
- **weak:** inyecciones de entrenamiento con SNR desde 3 en lugar de 5; 3 semillas.
- **wide:** el doble de canales; 3 semillas.

Las inyecciones de evaluación son las mismas para todas, con SNR de 3 a 50.

| Val | ROC-AUC | Recall con 1 % de falsas alarmas | Ruido que pasa con recall 95 % |
|---|---|---|---|
| base, una semilla (media ± desv.) | 0,820 ± 0,010 | 0,498 ± 0,024 | — |
| base, conjunto de 5 | 0,830 | 0,525 | 77 % |
| weak, conjunto de 3 | 0,824 | 0,520 | 79 % |
| wide, conjunto de 3 | 0,828 | 0,513 | 78 % |
| SDE + CNN plegada (línea base; estrellas BLS) | 0,877 | 0,473 | 67 % |
| **base + SDE + CNN plegada** | **0,889** | **0,564** | **66 %** |

Inyectados detectados con el umbral del 1 % (base, 5 semillas): SNR 10–15: 44 %; 20–30: 78 %; 30–50: 90 %.

**Conclusiones:**
- **Ninguna variante mejora.** La semilla 0 de la primera prueba (0,554) fue afortunada; la media de una semilla es 0,50.
- El conjunto de semillas da estabilidad, no mejora.
- Ni las inyecciones débiles ni la red ancha ayudan: la CNN sin plegar está en su techo con esta entrada.
- Con recall 95 %, la cadena combinada sigue dejando pasar el 66 % del ruido. Para señales con SNR ≥ 20, el ruido que pasa con recall 95 % es ~10 % (semilla 0).
- **Línea cerrada** (2026-11-06): el destino de la fase 5 pasa a ser el seguimiento terrestre (decisión 9). Si se retoma la búsqueda, el siguiente paso es un Transformer sobre esta misma entrada (`data/sample/series`).
