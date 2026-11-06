# Herramientas existentes para aficionados y para el vetting de TESS (fase 5, 2026-11-06)

Primera tarea de la fase 5: antes de construir los 3 modos, ver qué existe ya. Criterio fijado de antemano: si ya existe algo igual, se replantea antes de seguir. Las fuentes son búsquedas web del 2026-11-06 (enlaces al final); algunos detalles, como los criterios exactos de ExoClock, no se pudieron verificar.

## Lo que existe

| Herramienta | Qué hace | Para quién | Solapa con |
|---|---|---|---|
| **Swarthmore Transit Finder (Tapir)** | Tránsitos observables desde una ubicación y una fecha; código abierto | Cualquiera | Modo 1 (observabilidad) |
| **TESS Transit Finder** | Versión de Tapir para los TOIs; la usan los observadores del TFOP SG1 para elegir objetivos | Miembros del TFOP | Modo 1 |
| **NASA Exoplanet Archive, Transit and Ephemeris Service** | Tránsitos y visibilidad, con soporte de TOIs desde 2020 | Cualquiera | Modo 1 |
| **Prioridades del TFOP en ExoFOP** | Cada TOI tiene prioridad 1–5 por subgrupo (columnas `Master`, `SG1A`, `SG1B`, `SG2`…). De los 5 309 TOIs abiertos, 427 tienen prioridad `Master` 1 y 343 tienen `SG1B` 1 | TFOP | **Modo 1 (priorizar)** |
| **ExoClock (Ariel)** | Calendario personalizado según el telescopio y la ubicación, con priorización y alertas. Objetivo: refinar efemérides de ~1 000 planetas de Ariel; más del 40 % necesitaba actualización | Aficionados (~160 autores) | **Modo 1 (efeméride caducada)**, pero con planetas conocidos, no con candidatos |
| **Exoplanet Watch (NASA) + EXOTIC** | Listas de objetivos ordenadas (planetas conocidos, Ariel, candidatos de Planet Hunters TESS, TOIs) y reducción de imágenes a curvas | Aficionados | Modo 1 y modo 4 |
| **ExoMiner++** (NASA, 2025) | Clasificador de deep learning con catálogo público de puntuaciones para 147 568 señales de TESS a 2 min; 7 330 candidatos, de ellos 1 868 TOIs. Usa diagnósticos que nosotros no usamos (imagen diferencia, centroide…) y transferencia desde Kepler | Profesionales | **Modo 2 (probabilidad)**: mucho mejor que el nuestro (PR-AUC binario en TESS ~0,98 frente a 0,70) |
| **LEO-Vetter** (2025, código abierto, `pip`) | Vetting automático a nivel de flujo y de píxel, al estilo del Robovetter | Profesionales | Modo 2 |
| **TRICERATOPS** | Validación estadística: probabilidad de falso positivo | Profesionales | Modo 2 |
| **LATTE** | Inspección visual de curvas de TESS; la usa Planet Hunters TESS | Voluntarios | Modos 2 y 3 |
| **Planet Hunters TESS** (Zooniverse) | Búsqueda a ojo por voluntarios; ha encontrado candidatos que las tuberías pasaron por alto | Público | Modo 3 |
| **SPOC, QLP y TESS-ExoClass** | Búsqueda y vetting oficiales | Profesionales | Modo 3 |

Datos propios (ExoFOP, `data/labels/exofop_toi.csv`): hay 4 813 TOIs en PC; la mediana es Tmag 12,3 y profundidad 5 350 ppm. Unos **793** son asequibles para un aficionado (Tmag < 12 y profundidad > 3 000 ppm).

## Qué significa para nuestros modos

- **Modo 1 («¿qué observo?»):** casi todo existe ya.
  - La planificación la cubren Tapir, el Exoplanet Archive y ExoClock.
  - La priorización de TOIs la hace el propio TFOP, con información que nosotros no tenemos.
  - Lo único que no parece existir es juntarlo todo para el aficionado que no está en el TFOP, filtrando por lo que su telescopio puede detectar. Es una integración, no una novedad.
- **Modo 2 (informe):** en probabilidad, ExoMiner++ ya publica puntuaciones mejores que las nuestras para la mayoría de los TOIs a 2 min. Nuestro informe solo aportaría si es **didáctico y explicable** para el aficionado. No debería competir en exactitud.
- **Modo 3 (búsqueda):** las tuberías oficiales y Planet Hunters TESS lo cubren, y nuestra búsqueda es más débil (`research/12`).
- **Lo que sí es nuestro:**
  1. **Portabilidad demostrada**: un modelo solo de curva, entrenado con Kepler y K2, que se mide en un telescopio que no ha visto (LOMO). ExoMiner++ necesita diagnósticos propios de cada misión.
  2. **Evaluación honesta y prospectiva**: comparar a ciegas nuestras predicciones, las de ExoMiner++ y las prioridades del TFOP con lo que se vaya confirmando.
  3. **Explicación para no expertos, en español.**

## Conclusión

**El criterio se cumple: hay que replantear.** Construir los 3 modos tal como estaban sería en gran parte rehacer herramientas que ya existen, y con un modelo peor que ExoMiner++ en TESS. Las opciones están en la tarea de Notion (decisión de Jairo).

## Fuentes

- [Swarthmore Transit Finder](https://astro.swarthmore.edu/transits/)
- [NASA Exoplanet Archive, Transit and Ephemeris Service](https://exoplanetarchive.ipac.caltech.edu/docs/transit/transit_home.html)
- [TESS Follow-up Program (HEASARC)](https://heasarc.gsfc.nasa.gov/docs/tess/tfop.html) y [TESS Follow-up (MIT)](https://tess.mit.edu/followup)
- [ExoClock, arXiv:2012.07478](https://arxiv.org/pdf/2012.07478)
- [Exoplanet Watch: listas de objetivos](https://science.nasa.gov/citizen-science/exoplanet-watch/target-lists/)
- [ExoMiner++, arXiv:2502.09790](https://arxiv.org/abs/2502.09790)
- [LEO-Vetter, arXiv:2509.10619](https://arxiv.org/abs/2509.10619)
- [LATTE](https://pypi.org/project/tessLATTE/0.1.14)
- [Planet Hunters TESS (BAA)](https://britastro.org/wp-content/uploads/2019/06/Planet-hunters-TESS.pdf)
- [Herramientas de análisis de TESS (HEASARC)](https://heasarc.gsfc.nasa.gov/docs/tess/data-analysis-tools.html)
