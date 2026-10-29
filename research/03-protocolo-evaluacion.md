# Protocolo de evaluación (LOMO anidado)

Decidido el 2026-10-30 (tarea #34, opción B). Sustituye al protocolo de #24, que usaba la validación de la misión excluida para desarrollar y por eso la dejaba de hacer "no vista".

## Objetivo del protocolo

Medir si un modelo entrenado con unas misiones funciona en otra que no ha visto, sin que ninguna decisión (modelo, hiperparámetros, umbral, preprocesado) haya mirado esa misión.

## Definiciones

- **Misiones:** Kepler, K2 y TESS. CoRoT se añade como cuarta misión solo si llegan sus etiquetas (ver "CoRoT").
- **Clases:** PLANET, FALSE_POSITIVE y NO_SIGNAL (negativos reales, tareas #30 y #32).
- **Unidad:** la estrella (`star_id`). Ninguna estrella aparece en dos partes de un mismo experimento. Las estrellas compartidas entre misiones (cruce por coordenadas < 2", tarea #36) se excluyen del entrenamiento cuando su otra misión es la excluida.
- **Splits del dataset** (`train`, `val`, `test`): se conservan tal cual vienen, por estrella.

## Fold externo: la misión excluida

Para cada misión M en {Kepler, K2, TESS}:

1. M es la **misión excluida**. No participa en ningún paso de ajuste: ni normalización, ni selección de variables, ni hiperparámetros, ni umbral.
2. Las otras dos misiones (A y B) forman el **conjunto de entrenamiento**.
3. Se elige el modelo con el **bucle interno** (abajo), usando solo A y B.
4. Se reentrena el modelo elegido con `train` y `val` de A y B.
5. Se evalúa **una sola vez** en M. Como M no se usó para nada, se evalúa en M entera (`train`, `val` y `test` de esa misión), lo que da más estrellas por clase, sobre todo en K2.

## Bucle interno: elegir sin mirar M

Dentro de cada fold externo hay dos folds internos, que son transferencias entre las dos misiones restantes:

- entrenar en `train` de A y validar en `val` de B,
- entrenar en `train` de B y validar en `val` de A.

El criterio de selección es la media de PR-AUC de los dos folds internos. Aquí se eligen la familia de modelo, los hiperparámetros y el umbral (regla pendiente, ver "Pendiente"). El preprocesado que aprende parámetros (escaladores, imputación) se ajusta solo con los datos de entrenamiento de cada fold.

## Evaluación dentro de dominio

Aparte, una sola vez al final: modelo entrenado con `train` y `val` de una misión y evaluado en el `test` de **esa misma misión**. Sirve de referencia para medir la brecha de generalización (rendimiento fuera de dominio frente a dentro).

## Reglas

- La **familia de modelos se fija con los folds internos**. Si después de ver los resultados externos se cambia de familia o de preprocesado, esos resultados pasan a considerarse exploratorios y se anota la razón en el registro de decisiones.
- **Línea base en cada fold:** dummy (clase más frecuente en las misiones de entrenamiento).
- **Métricas:** PR-AUC por clase y por misión, macro-F1, y los puntos de la tarea #37 (calibración y recall a precisión fija). Intervalos de confianza al 95 % con bootstrap agrupado por estrella.
- **Resultado a informar:** una tabla con tres filas externas (una por misión excluida), la fila dentro de dominio y la brecha.
- **Modelo final:** una vez cerrado el protocolo, se entrena con las tres misiones para el uso real. Su rendimiento esperado es el de los folds externos, no el de su propio entrenamiento.
- Toda aleatoriedad lleva semilla fija.

## CoRoT

Si se consiguen las etiquetas (tarea #31), CoRoT se evalúa como misión excluida adicional, **sin participar nunca en la selección** (opción A). Es la prueba más limpia de portabilidad, con una muestra pequeña e intervalos de confianza amplios.

## Pendiente

- **Regla del umbral en una misión sin etiquetas.** El prior de falsos positivos cambia mucho entre misiones (Kepler 56 %, K2 16 %, TESS 18 %) y un umbral fijado en A y B no se transfiere. Opciones a decidir: umbral del bucle interno, ajuste por prior, o informar solo métricas sin umbral (PR-AUC) y el recall a precisión fija.
- **Utilidad de folds** (`lomo_folds`) y baseline dummy: el trabajo registrado en la tarea #27 no existe en el repo, hay que rehacerlo.
