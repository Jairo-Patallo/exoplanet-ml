# Armonización de etiquetas entre misiones

Tarea #33, 2026-10-30. Código: `src/exoplanet/labels.py` (mapeo y fuerza), `scripts/fetch_labels.py` (descarga las tablas originales) y `scripts/label_audit.py` (auditoría). Tests: `tests/test_labels.py`.

## Problema

PLANET y FALSE_POSITIVE del dataset de Hugging Face unen criterios distintos según la misión, y el dataset no conserva las disposiciones originales de TESS y K2. Se cruzaron con el NASA Exoplanet Archive (tablas `toi` y `k2pandc`), eligiendo el candidato de periodo más cercano cuando una estrella tiene varios.

## Qué hay en cada misión

| Misión | Clase del dataset | Disposición original | n | Fuerza |
|---|---|---|---|---|
| Kepler | PLANET | confirmado (`kepler_confirmed`) | 1 943 | fuerte |
| Kepler | PLANET | candidato del Robovetter (`kepler_robovetter`) | 1 631 | débil |
| Kepler | FALSE_POSITIVE | Robovetter | 4 638 | fuerte |
| Kepler | FALSE_POSITIVE | `kepler_confirmed` (contradictorio) | 2 | excluida |
| TESS | PLANET | CP (confirmado) | 674 | fuerte |
| TESS | PLANET | PC (candidato) | 3 802 | débil |
| TESS | PLANET | APC (candidato ambiguo) | 438 | excluida |
| TESS | PLANET | FP o FA en el archivo actual | 58 | excluida (conflicto) |
| TESS | FALSE_POSITIVE | FP | 968 | fuerte |
| TESS | FALSE_POSITIVE | FA (falsa alarma) | 87 | excluida |
| TESS | FALSE_POSITIVE | PC en el archivo actual | 1 | excluida (conflicto) |
| K2 | PLANET | CONFIRMED | 346 | fuerte |
| K2 | PLANET | CANDIDATE | 899 | débil |
| K2 | FALSE_POSITIVE | FALSE POSITIVE | 240 | fuerte |

Cualquier combinación que no esté en la tabla es `excluida`. NO_SIGNAL no entra aquí: sus negativos se generan aparte (`03-protocolo-evaluacion.md`).

## Reglas

- **Test y evaluación:** solo etiquetas **fuertes**. La métrica final no se contamina con ruido de etiquetas.
- **Entrenamiento:** dos variantes, solo fuertes frente a fuertes más débiles. Se elige con los folds internos del protocolo, evaluando siempre en etiquetas fuertes para que sean comparables.
- **Excluidas:** no se usan en ningún conjunto. Las FA de TESS (87) quedan anotadas como posible complemento de NO_SIGNAL "con estructura" (sistemáticos reales).
- **Discrepancias con el archivo actual** (58 filas PLANET de TESS que el archivo hoy llama FP o FA, y 1 FP que hoy es PC): el dataset es una instantánea antigua y las disposiciones cambian. Se excluyen en lugar de decidir cuál es la correcta.

## Consecuencias medibles

- **La prevalencia de falsos positivos cambia mucho** al quedarse solo con etiquetas fuertes: Kepler 56,5 % → 70,5 %, K2 16,2 % → 41,0 %, TESS 17,5 % → 59,0 %. El listón del dummy cambia (`results/baseline/dummy.csv`, variante "solo etiquetas fuertes").
- **El test se encoge.** Dentro de dominio, el test fuerte tiene 656 estrellas en Kepler, 167 en TESS y solo **62 en K2**. Para K2 conviene apoyarse en el fold externo (la misión entera, 586 estrellas) y asumir intervalos anchos.
- **Pendiente de medir con modelos:** comparar entrenar con solo fuertes frente a fuertes más débiles. Es parte de la selección en los folds internos desde el Escalón 3.

## Cómo reproducirlo

```
python scripts/fetch_labels.py   # descarga toi.csv y k2pandc.csv a data/labels/ (ignorado por git)
python scripts/label_audit.py    # results/labels/audit.csv y strength_counts.csv
```
