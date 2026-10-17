# Como vamos a abordar el problema?

Primeras cosas que hay que tener en cuenta. Lo primero, el descenso de brillo de una estrella cuando un planeta está en tránsito.
Para una estrella de Radio R* y un planeta de radio R_p, la profundidad de tránsito es aproximadamente:

(R_p / R*)**2

# Primeros test

Se ha utilizado una base de datos de código abierto de HuggingFace llamada exoplanet-transit-detection, para más detalles con los nombres de las variables,
se puede consultar en la propia página del dataset ---- https://huggingface.co/datasets/bingbangboom/exoplanet-transit-detection

Para hacer el primer baseline de este proyecto primero tendremos que hacer un lavado de datos. Hemos eliminado diferentes datos de nuestro database por diversos motivos.

Fuga de etiquetas: label_source, confidence_score, kepoi_name, disposition_disputed, fp_flag_* .
Identificadores de la estrella: star_id, ra, dec
Atajos: mission, cadence, has_transit_params, n_planets_in_system, period_days, epoch_bjd, duration_hrs, depth_ppm, planet_radius_earth, flux_*_len.
Esto es porque los datos que están etiquetados como NO_SIGNAL tienen estos datos vacíos, por tanto, solo con mirar si la columna está vacía, el modelo podría adivinar que no tienen señal.

Quedan las variables teff, logg, radius, mass, metallicity, kepmag, cdpp_3hr. 
Las curvas se usarán aparte como entrada del modelo.