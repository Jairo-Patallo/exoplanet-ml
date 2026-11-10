# TÉCNICAS DE MACHINE LEARNING Y DEEP LEARNING PARA EL DESCUBRIMIENTO DE EXOPLANETAS A TRAVÉS DE TRÁNSITOS

Los exoplanetas son planetas que orbitan alrededor de otras estrellas fuera de nuestro sistema solar,
están ocultos debido al resplandor brillante de las estrellas a las que orbitan, por tanto, los astrónomos utilizan
diferentes técnicas para detectarlos.

Una forma para detectar exoplanetas es buscar estrellas tambaleantes. Una estrella que tiene planetas no orbita perfectamente alrededor de su centro,
esta órbita esta descentrada y hace que la estrella parezca una estrella tambaleante. Con este método se han detectado cientos de exoplanetas,  el  método  de  tránsito  es  el  más  productivo  (≈  3/4  de  los  planetas  conocidos)
gracias a misiones espaciales que miden con precisión de decenas de ppm

### ¿Cómo podemos detectar planetas similares a la tierra en otros sistemas solares?

Para la detección de exoplanetas lo más recomendable es utilizar técnicas de Tránsito, esta técnica es mi base y sobre la que me apoyaré para hacer los experimentos.

### ¿Cómo funciona el método de Tránsito para la detección de exoplanetas?

Un tránsito de un exoplaneta ocurre cuando el planeta se interpone entre el Telescópio y la estrella a la cual orbita. Hay una disminución de luz en la estrella lo cual nos permite conocer
parámetros orbitales y algunas características físicas del planeta que utilizando otras técnicas sería imposible. 
Ahora cómo se detectan. Los científicos observan las estrellas y hacen gráficos llamados curvas de luz. Se basan en la cantidad de luz de una estrella que llega al telescopio, hay que fijarse
en las pequeñas reducciones de brillo  de una estrella. Un  candidato  a  tránsito  es  una  señal  que  se  repite  con  periodo  P  constante (normalmente se exigen ≥ 2-3 tránsitos),
con profundidad y duración consistentes entre eventos y con una forma compatible con un disco opaco pequeño cruzando un disco estelar con oscurecimiento del limbo.

El problema principal es que cualquier evento en la superficie de la estrella que la acerque hacia el observador cambia la cantidad de luz por tanto hay que saber detectarlos. Las estrellas tienen ondas que se propagan por su superficie
haciendo que la estrella tiemble,. por tanto, pueden darse cambios de posición debido a esto. Variaciones en su campo magnético que produzcan flujos que pueden provocar variaciones en su velocidad radial, al final, todo esto
se refleja en variaciones de la superficie de la estrella en la dirección del observador. Por tanto, a corto o largo plazo se pueden confundir con la presencia de planetas rocosos, por tanto, nuestros sistemas deben tener una alta sensibilidad
para poder detectar todos los cambios y saber cuando existen cambios en la superficie de la estrella y no son un exoplaneta.

Una curva de luz representa cómo cambia el brillo observado de un objeto astronómico a lo largo del tiempo. Cuando un exoplaneta pasa por delante como ya dijimos la luz disminuye. Los telescopios obviamente no pueden ver estos exoplanetas a cientos de años luz, pero generan unas series temporales con brillo, cuando la luz total que entra por el telescopio disminuye
existe un posible exoplaneta lo que llamamos candidato, adjunto imágen.

![Tránsito de un exoplante](/wasp-10b.jpg)

### Falsos positivos

Lo más importante a la hora de realizar nuestra investigación es saber diferenciar la presencia de un exoplaneta de cualquier otro cambio de brillo en la estrella. Por ello, vamos a categorizar diferentes falsos positivos
Una de las cosas que distingue al resto de observaciones de un exoplaneta es la forma de la curva de luz, ya que hay diferentes formas de las curvas de luz cuando se tratan de otros objetos que pasan delante de la estrella observada,
por ejemplo, una estrella binaria eclipsante tendrá picos en forma de V, mientras que los de un exoplaneta tiene su curva menos marcada, siendo similar a una U. Un sistema binario es aquel en el que existen dos estrellas que orbitan entre ellas, por tanto, 
cuando una estrella pasa por delante de otra, se detectará una caída de luz mayor que la de un exoplaneta, mientras que el tránsito de un exoplaneta es mucho menor, debido principalmente a su tamaño. También hay que añadir
que la curva del exoplaneta tendrá una caída mucho mas rápida.
