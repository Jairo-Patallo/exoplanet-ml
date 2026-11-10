# Guía para redactar: qué herramientas existen ya

**Idea central de la sección:** antes de construir una herramienta propia revisaste lo que existía, y casi todo lo planeado ya estaba hecho (planificadores, priorización del TFOP, ExoMiner++, búsqueda oficial). Esto obligó a reorganizar el proyecto y a definir qué aporta de verdad: portabilidad demostrada, evaluación prospectiva y explicación en español para no expertos.

## Introducción: por qué y cómo

**Por qué se hizo.**
- Construir algo que ya existe es tiempo perdido, y además queda mal en un proyecto.
- La regla fijada de antemano fue: si existe algo igual, se replantea antes de seguir.

**Cómo se hizo.**
- Búsqueda web de herramientas para aficionados (planificación de observaciones) y para profesionales (vetting y búsqueda en TESS).
- Para cada una se anotó qué hace, para quién es y con qué parte del plan coincide.
- Se calculó con ExoFOP cuántos candidatos abiertos hay y cuántos son asequibles para un aficionado.

**Qué se buscaba conseguir.** Decidir qué construir y qué no, y encontrar el hueco real del proyecto.

## Qué contar, en orden

1. **El plan inicial:** recomendar qué observar, informar de un candidato y buscar en una estrella.
2. **Lo que existe,** agrupado:
   - **Planificar:** Tapir/Swarthmore, el servicio del Exoplanet Archive, ExoClock (Ariel) y Exoplanet Watch con EXOTIC.
   - **Priorizar:** las prioridades del TFOP en ExoFOP (de unos 5 300 TOIs abiertos, 427 con prioridad máxima).
   - **Clasificar:** ExoMiner++ (catálogo de unas 150 000 señales), LEO-Vetter y TRICERATOPS.
   - **Buscar e inspeccionar:** SPOC, QLP, Planet Hunters TESS y LATTE.
3. **El dato propio:** 4 813 TOIs sin resolver; unos 793 asequibles para un aficionado (Tmag < 12 y profundidad > 3 000 ppm).
4. **La conclusión:**
   - Recomendar ya existe; integrarlo para aficionados sería solo una integración.
   - Informar con probabilidades no aporta frente a ExoMiner++ (PR-AUC ~0,98 frente a 0,70–0,77).
   - Buscar ya lo hacen mejor otros.
5. **Lo que sí es tuyo,** los tres puntos:
   - portabilidad demostrada (un modelo solo de curva medido en telescopios no vistos);
   - evaluación prospectiva frente a ExoMiner++ y el TFOP;
   - explicación en español para no expertos.
6. **La decisión:** reorganización en fases 6–9 (mejora de TESS, investigación, herramienta explicativa, web y documentación). Menciona la posible colaboración con DIPS Ojos (Universidad de Oviedo y Sociedad Astronómica Asturiana Omega).

## Conceptos que conviene explicar al lector

- **TOI:** candidato de TESS.
- **TFOP:** programa de seguimiento de TESS.
- **ExoFOP:** la plataforma donde se comparten los candidatos y sus prioridades.
- **Pro-am:** colaboración entre profesionales y aficionados.

## Datos clave para citar

4 813 TOIs abiertos · 793 asequibles · 427 con prioridad TFOP máxima · ExoMiner++ con ~147 568 señales puntuadas.

## Figuras

Esta sección no necesita una gráfica: funciona mejor con una **tabla** de herramientas (columnas: qué hace, para quién, con qué parte del plan coincide).

## Limitaciones que mencionar

- La revisión se hizo con búsquedas web; algunos detalles, como los criterios de ExoClock, no se pudieron verificar.

## Fuentes

- Swarthmore Transit Finder: https://astro.swarthmore.edu/transits/
- Exoplanet Archive, tránsitos y efemérides: https://exoplanetarchive.ipac.caltech.edu/docs/transit/transit_home.html
- TFOP: https://heasarc.gsfc.nasa.gov/docs/tess/tfop.html y https://tess.mit.edu/followup
- ExoFOP-TESS: https://exofop.ipac.caltech.edu/tess/
- ExoClock (Kokori et al.): https://arxiv.org/abs/2012.07478
- Exoplanet Watch: https://science.nasa.gov/citizen-science/exoplanet-watch/target-lists/
- ExoMiner++: https://arxiv.org/abs/2502.09790
- LEO-Vetter: https://arxiv.org/abs/2509.10619
- TRICERATOPS (Giacalone et al. 2021): https://arxiv.org/abs/2002.00691
- LATTE: https://pypi.org/project/tessLATTE/0.1.14
- Planet Hunters TESS (BAA): https://britastro.org/wp-content/uploads/2019/06/Planet-hunters-TESS.pdf
- Herramientas de TESS (HEASARC): https://heasarc.gsfc.nasa.gov/docs/tess/data-analysis-tools.html
- DIPS Ojos: https://dipsojos.uniovi.es
