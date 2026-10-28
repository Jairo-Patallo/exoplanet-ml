# Análisis exploratorio (Fase 1 + exploración de variables de la Fase 2)

Código: `scripts/eda.py` (análisis) y `src/exoplanet/features.py` (variables físicas). Ejecutar: `python scripts/eda.py`.
Salida completa: `results/eda/eda_output.txt`; figuras y tablas en `results/eda/`.

**Regla seguida:** solo se explora **train** (18 853 estrellas). **Val** se usa únicamente en el test de atajos
(entrenar en train → medir en val). **Test no se ha tocado.**

## 0. Decisiones tomadas a partir de la hoja de ruta

| Decisión | Por qué |
|---|---|
| Analizar sobre todo la tarea de **vetting** (PLANET vs FP, 12 250 filas con vistas plegadas) | Fase 0: es el problema científico; NO_SIGNAL tiene vistas a cero (Problema B) |
| Separabilidad con **AUC de Mann-Whitney** + IC 95 % bootstrap | No depende de umbral ni de la escala (es de rangos); AUC = P(X_planeta > X_fp) |
| Todo **estratificado por misión** | La misión es un factor de confusión (Problema A); hay que comprobar el efecto Simpson |
| Spearman en lugar de Pearson | Variables con colas de varios órdenes de magnitud; Spearman solo usa rangos |
| PCA sobre la **forma** normalizada `(1-f)/δ` | Sin normalizar, la PCA solo vería la profundidad, que ya tenemos como variable |

## 1. Integridad

- No hay periodos ≤ 0 ni profundidades ≤ 0. Sí hay 5 duraciones ≤ 0, 2 profundidades ≥ 100 % y 1 duración > periodo: errores de catálogo.
- **28 PLANET con Rp de catálogo > 100 R⊕** (17 del Robovetter, 10 de K2). Físicamente imposible para un planeta (Júpiter ≈ 11 R⊕):
  es **ruido de etiqueta** (Problema D) o un radio estelar erróneo.
- 331 filas PLANET/FP no tienen vistas plegadas (282 PLANET y 49 FP): se excluyen del vetting.
- **Artefacto descubierto:** los bins 0-4 y 76-80 de `flux_local` tienen una dispersión 2-3 veces mayor que el resto de la zona
  fuera del tránsito. En la primera PCA, PC1 (67 % de la varianza) era *solo* ese borde. Ahora se excluyen (`LOC_VALID`).
  **Consecuencia para la CNN:** recortar esos bins o esperar que la red aprenda ruido de borde.

## 2. Valores faltantes: el atajo está confirmado

% de NaN (train): en TESS faltan `mass`, `metallicity` y `cdpp_3hr` el **100 %** de las veces; en K2 falta `cdpp_3hr` el 100 %.
En PLANET, `mass` falta el 59 % de las veces; en FP, el 23 %.

## 3. Test de atajos (macro-F1 en val)

| Tarea | Dummy | Solo `isna()` estelar | Solo misión |
|---|---|---|---|
| 3 clases | 0.196 | **0.478** | 0.458 |
| Vetting | 0.384 | **0.687** | **0.687** |

Un modelo que **no ve ni un valor físico**, solo *qué falta*, sube el macro-F1 de 0.38 a 0.69 en vetting. Que la máscara
y la misión den el mismo resultado demuestra que **la máscara de NaN es la misión encubierta**.
Cualquier modelo futuro debe superar claramente 0.687 para demostrar que aprende algo más que la misión.

## 4. Separabilidad univariante (AUC, vetting)

Lectura: 0.5 = no separa; < 0.5 = los PLANET tienen valores menores. Selección (tabla completa en `04_auc_by_mission.csv`):

| Variable | Todas | Kepler | K2 | TESS |
|---|---|---|---|---|
| depth_ppm | **0.50** | 0.40 | **0.07** | **0.59** |
| rp_implied | 0.47 | 0.37 | 0.08 | 0.53 |
| kepmag | 0.38 | 0.53 | 0.41 | 0.64 |
| duty_cycle (T/P) | 0.36 | 0.32 | 0.19 | 0.36 |
| ingress_frac (τ/T) | 0.34 | 0.31 | 0.38 | 0.41 |
| snr_expected | 0.35 | 0.35 | — | — |
| metallicity | 0.66 | 0.66 | 0.62 | — |

Conclusiones:
1. **Efecto Simpson en la profundidad.** En el conjunto global no separa (AUC 0.50), pero dentro de cada misión sí, y **con
   signos opuestos**: en Kepler y K2 los FP son más profundos; en TESS ocurre lo contrario. Un modelo sin estratificar
   aprendería una relación que no existe en ninguna misión por separado.
2. **`kepmag` es un atajo:** en el global da 0.38, pero en Kepler 0.53. Casi toda su señal viene de que las estrellas de TESS son más brillantes.
3. Las variables **robustas entre misiones** (mismo signo en las tres) son `duty_cycle`, `ingress_frac`, `teff`
   y `period_days`. Son buenas candidatas para generalizar.
4. `metallicity` (0.66) no puede usarse tal cual: falta en TESS y su tasa de NaN depende de la clase.

## 5. Reglas físicas (escalón 2 de la escalera; positivo = FP; precisión del azar = 0.384)

| Regla | % marcado | Precisión FP | Recall FP |
|---|---|---|---|
| δ > 2 % (regla original) | 12 % | 0.75 | 0.24 |
| Rp implícito > 22 R⊕ | 10 % | **0.86** | 0.22 |
| ρ_tránsito/ρ_catálogo < 0.1 | 4 % | **0.94** | 0.11 |
| Forma V (τ/T > 0.35, SNR > 10) | 19 % | 0.72 | **0.36** |
| Par/impar z > 3 | 19 % | 0.39 | 0.19 |
| Secundario z > 3 | 28 % | 0.46 | 0.33 |

- El criterio **Rp = R*·√δ** supera a la regla del 2 % en precisión (0.86 frente a 0.75) con un recall parecido. Confirma el Error 9 del informe.
- **Densidad estelar (ec. 6):** con b = 0, ρ_tr es una **cota superior** de la densidad real, porque un b > 0 acorta T
  y obliga a una a/R* menor. Por eso, un ρ_tr diez veces menor que el del catálogo no se puede explicar con la geometría:
  el 94 % de los casos marcados son FP. Es la regla más precisa, aunque de bajo recall.
- **Forma (τ/T):** para un trapecio simétrico, el déficit medio en T es δ(1-τ/T), así que τ/T = 1 - déficit_medio/δ.
  Los planetas salen con τ/T ≈ 0.25, no 0.09 como en la teoría, porque el binning (T ≈ 10 bins) y el oscurecimiento
  del limbo suavizan los bordes. El valor absoluto está sesgado, pero **ordena bien** (AUC 0.34).

### Advertencia: los tests z par/impar y secundario no están calibrados
Si el ruido fuera gaussiano e independiente entre bins, en los PLANET se cumpliría |z| ~ |N(0,1)|: mediana 0.67 y P(|z|>3) = 0.003.
Lo medido es:
- par/impar: mediana 1.16, P(|z|>3) = **0.197**
- secundario: mediana 2.60, P(|z|>3) = **0.449**

Por tanto **la hipótesis de bins independientes es falsa** en las vistas de 201 bins: están suavizadas y hay ruido rojo,
así que σ/√n subestima el error. Estos z **no son significancias** y no se puede usar "z > 3" como test.
Siguen sirviendo como variables de ranking. Arreglo futuro: estimar el error con la varianza entre bloques
(*block bootstrap*) en lugar de σ/√n.

## 6. Redundancia (Spearman |ρ| > 0.7)

`logg`–`radius` (-0.97), `depth_ppm`–`depth_local` (0.95), `planet_radius_earth`–`rp_implied` (0.92),
`period`–`duty_cycle` (-0.89), `kepmag`–`cdpp_3hr` (0.81). Para la regresión logística conviene quedarse con una de cada par.
**Comprobación cruzada:** la profundidad medida en la curva coincide con la del catálogo (ρ = 0.95) y el Rp implícito
con el del catálogo (mediana |log10 ratio| = 0.036, es decir, ~8 %). Los datos son coherentes entre sí.

## 7. Forma de las curvas y PCA (flux_local, SNR > 10, n = 9 870)

- Tras normalizar por δ, la mediana de los FP tiene el fondo más puntiagudo (V) y la de los PLANET es más plana (U).
- PCA: PC1 (44 %) recoge el nivel fuera del tránsito y PC2/PC3 (10 % cada una) el ancho y la asimetría del egreso y el ingreso.
  Ninguna PC separa bien por sí sola (AUC 0.56-0.62). La diferencia de forma existe pero es sutil y **no lineal**.
  Esto justifica probar luego una CNN (escalones 5-6).

## Implicaciones para la Fase 3

1. **Listón real** para vetting: macro-F1 de val **0.687**, el que se consigue solo con la misión, no el 0.384 del dummy.
2. Reportar siempre las **métricas por misión**. Alternativa: entrenar solo con Kepler para quitar el confusor.
3. No usar `kepmag`, `metallicity`, `mass` ni `cdpp_3hr` sin controlar la misión.
4. Variables candidatas para la regresión logística: `log period`, `log duty_cycle`, `ingress_frac`, `log rp_implied`,
   `log_rho_ratio`, `teff`, `logg`, `odd_even_z`, `secondary_z` (estas dos solo como ranking).
5. Para la CNN, recortar los bins de borde de `flux_local`.
6. Limpiar o ponderar los 28 PLANET con Rp > 100 R⊕ (ruido de etiqueta).
