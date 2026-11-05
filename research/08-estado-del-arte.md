# Estado del arte: clasificadores de tránsitos con redes neuronales (#38, fase 4)

Revisión del 2026-11-05. Cifras de los artículos (leídas en arXiv/ar5iv; las marcadas con * salen de un resumen automático de las tablas y conviene comprobarlas en el PDF antes de citarlas en la memoria).

## 1. Trabajos principales

| Trabajo | Misión | Tarea | Datos y protocolo | Entrada | Resultado publicado |
|---|---|---|---|---|---|
| **AstroNet** (Shallue & Vanderburg 2018, AJ 155, 94) | Kepler DR24 | Binaria: PC frente a AFP + NTP | 15 737 TCE (3 600 PC). Partición **al azar por TCE** 80/10/10; test 1 523 | Global 2001 bins + local 201 bins, solo flujo | Exactitud 0,960*, **ROC-AUC 0,988**; recall 0,95 a precisión 0,90 |
| **AstroNet-K2** (Dattilo et al. 2019, AJ 157, 169) | K2 (C0–C16) | Binaria: candidato frente a binaria + basura | 27 634 TCE de un BLS propio, etiquetados a mano (~95 % negativos*). Al azar 80/10/10; test 2 755 | Global 701 + local 51 bins, + Rp/R* y parámetro de impacto | Exactitud 97,8 %, **ROC-AUC 0,988**. A precisión 1, recall 0,2*: "necesita supervisión humana" |
| **AstroNet-Triage / Vetting** (Yu et al. 2019, AJ 158, 25) | TESS (QLP, FFI) | Triage: tránsito frente a ruido/variabilidad. Vetting: planeta frente a binaria | TCE de QLP etiquetados a mano | Vistas de AstroNet + cambio de profundidad y secundario | Triage: AP 0,970, exactitud 0,974. **Vetting: AP 0,693**, exactitud 0,978 |
| **ExoMiner** (Valizadegan et al. 2022, ApJ 926, 120) | Kepler DR25 | Binaria: PC frente a AFP + NTP | 30 609 TCE (2 291 PC, 24 779 NTP). **10 folds por estrella** | Flujo (301/31 bins), centroide, par/impar, secundario + escalares (parámetros estelares, desplazamiento del centroide, diagnósticos DV) | Precisión 0,968, recall 0,974, **PR-AUC 0,995**, ROC-AUC 1,000*; recall 0,936 a precisión 0,99. AstroNet reentrenado en el mismo protocolo: PR-AUC 0,925* |
| **ExoMiner++** (Valizadegan et al. 2025, arXiv 2502.09790) | TESS SPOC 2 min (+ Kepler en el entrenamiento) | Binaria: planeta (KP + CP) frente a BD, EB, FP, NTP; sin PC | 57 162 TCE (6,4 % planetas). 10 folds por estrella; val y test solo TESS | Todo lo de ExoMiner + imagen de diferencia, periodograma, flujo sin plegar, datos de actitud | Precisión 0,933, recall 0,951, **PR-AUC 0,976**, ROC-AUC 0,998. Añadir Kepler al entrenamiento: PR-AUC 0,967 → 0,976 |

**Transferencia sin reentrenar entre telescopios** (nuestro LOMO): apenas está estudiada. ExoMiner++ entrena siempre con TESS (y ve mejor juntar Kepler + TESS que el ajuste fino); no publica la cifra de Kepler aplicado directamente a TESS. Hay indicios aislados: un preprint de 2026 (EXOVEIL) detecta 47 sistemas de TESS con un modelo de Kepler sin reentrenar, y otro trabajo de 2026 ("Depth-Blind by Design") avisa de que un clasificador de Kepler congelado aplicado a TESS se fija en artefactos propios de TESS (el periodo orbital de 13,7 d). Ansdell et al. (2018) ya sugerían que los modelos pequeños generalizan mejor de Kepler a TESS. **No hemos encontrado una evaluación LOMO de tres telescopios como la nuestra**; la búsqueda no fue exhaustiva.

## 2. Nuestros resultados en el mismo formato (LOMO, sin ver el telescopio evaluado)

`results/models/lomo_cnn_fixed_pred.csv`, probabilidad de PLANET como puntuación.

| Misión excluida | PLANET frente al resto: ROC-AUC / PR-AUC (prevalencia) | PLANET frente a FP (como el "vetting"): ROC-AUC / PR-AUC (prevalencia) |
|---|---|---|
| Kepler (entrenado con K2 + TESS, 3 280 estrellas) | 0,965 / 0,831 (0,17) | 0,927 / 0,841 (0,33) |
| K2 (entrenado con Kepler + TESS) | 0,961 / 0,926 (0,26) | 0,942 / 0,948 (0,51) |
| TESS (entrenado con Kepler + K2) | 0,895 / 0,691 (0,20) | 0,778 / 0,704 (0,39) |

## 3. Por qué las cifras no se pueden comparar directamente

1. **Protocolo.** Todos los trabajos publicados entrenan y prueban en el mismo telescopio. El nuestro prueba en un telescopio que el modelo no ha visto; nuestro dentro de misión equivalente (fase 2) es más alto (PR-AUC macro 0,94 en Kepler, 0,96 en K2).
2. **Negativos.** Su clase negativa está dominada por NTP (ruido, artefactos y variabilidad: 81 % en ExoMiner), que son fáciles y suben la ROC-AUC. Nuestros NO_SIGNAL son estrellas sin señal plegadas con una efeméride al azar, y la clase difícil (FP) pesa mucho más.
3. **Partición.** AstroNet y AstroNet-K2 parten **al azar por TCE**: varias señales de la misma estrella pueden quedar en train y en test (fuga). ExoMiner y nosotros partimos por estrella.
4. **Entrada.** ExoMiner y ExoMiner++ usan centroide, imagen de diferencia y parámetros estelares, que resuelven buena parte de los FP que nosotros no vemos (techo físico de la fase 2). Nosotros usamos solo la curva, a propósito, para ser portables (CoRoT no tiene centroide).
5. **Efeméride.** Todos parten de la efeméride de un pipeline de detección (TCE de Kepler, SPOC, QLP o un BLS propio). Nosotros usamos la del catálogo: es la carencia que resuelve #53 (BLS).

## 4. Qué referencia externa usaremos

- **Referencia de arquitectura: AstroNet.** Nuestra CNN es la suya (dos ramas global/local, solo flujo). Es la comparación justa en cuanto a información de entrada. Referencia: ROC-AUC 0,988 en Kepler (partición al azar, dentro de misión) y PR-AUC 0,925 reentrenado por estrella en el protocolo de ExoMiner.
- **Referencia de TESS: AstroNet-Vetting** (planeta frente a binaria en TESS, entrenado con TESS): AP 0,693. Nuestro PLANET frente a FP en TESS **sin haber visto TESS** da 0,704. No son los mismos datos (QLP/FFI frente a SPOC 2 min) ni las mismas etiquetas, pero indica que el LOMO de TESS está al nivel de un clasificador de 2019 entrenado en el propio TESS.
- **Techo con más información: ExoMiner / ExoMiner++** (PR-AUC 0,995 en Kepler, 0,976 en TESS). Sirve para cuantificar en la model card cuánto aporta lo que nosotros no usamos (centroide, imagen de diferencia, parámetros estelares).
- **En la memoria:** comparar con estas cifras solo de forma cualitativa y con los cinco avisos de la sección 3. La comparación cuantitativa honesta sería reentrenar AstroNet con nuestro protocolo, pero eso ya es nuestra CNN de la fase 2.

## 5. Lecciones para la fase 4

- **BLS (#53).** AstroNet-K2 genera sus TCE con un BLS propio, y su clase negativa (basura) son las detecciones espurias de ese BLS. Con BLS nuestro modelo verá señales que el catálogo nunca tuvo (alias del periodo, artefactos): habrá que medir el % de recuperación de la efeméride y, probablemente, entrenar o al menos evaluar con vistas plegadas con la efeméride de BLS.
- **Bins por misión.** AstroNet-K2 bajó a 701/51 bins porque las curvas de K2 son más cortas. Nuestras 201/81 bins son comunes a las tres misiones: es una elección coherente con la portabilidad.
- **Ensemble.** AstroNet-K2 y ExoMiner++ promedian 10 modelos; nosotros, 3 semillas. Subir a 5–10 es barato si se quiere el último punto.
- **Juntar misiones mejor que ajustar fino** (ExoMiner++): coincide con nuestra decisión de entrenar el modelo final con Kepler + K2 + TESS juntos.
- **Alta pureza.** Ni AstroNet-K2 (recall 0,2 a precisión 1) ni nosotros en TESS (recall 0,12 a precisión 0,9) sirven para validar planetas solos: el uso honesto es priorizar candidatos. ExoMiner llega a validar porque usa diagnósticos fuera de la curva.

## Fuentes

- Shallue & Vanderburg (2018): https://arxiv.org/abs/1712.05044
- Dattilo et al. (2019): https://arxiv.org/abs/1903.10507
- Yu et al. (2019): https://arxiv.org/abs/1904.02726
- Valizadegan et al. (2022), ExoMiner: https://arxiv.org/abs/2111.10009
- Valizadegan et al. (2025), ExoMiner++: https://arxiv.org/abs/2502.09790
- Ansdell et al. (2018), Exonet: https://arxiv.org/abs/1810.13434
- Tey et al. (2023), AstroNet V (triage TESS): citado en los resultados de búsqueda; no leído
- EXOVEIL (2026): https://arxiv.org/abs/2606.02778 ; "Depth-Blind by Design" (NeurIPS 2026): https://neurips.cc/virtual/2026/164256 (solo resúmenes, sin leer el texto completo)
