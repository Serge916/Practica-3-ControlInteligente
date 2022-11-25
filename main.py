import matplotlib.pyplot as plt
import localizacion as loc
import plots

dist = 200    # (m) este parámetro no se cambia
t_max = 5000  # (s) este parámetro no se cambia

#seeds = [123456789, 234567891, 34567891, 456789123, 567891234, 67891234, 7891234, 89123456, 912345678, 213456789]
seeds = [123456789]

q = 0.2 ** 2  # prueba con distinto ruido odométrico, varianza (m2)
r = 0.2 ** 2  # prueba con distinto ruido sensorial, varianza (m2)
rango_sensor = 5  # prueba con distinto alcance sensorial (m)

# Localización por odometría
estados1 = loc.localizacion_odometrica(q, r, rango_sensor, dist, t_max, seeds[0])
plots.plot_estados(estados1, "localizacion_odometrica.pdf", dist)

# Localización con FK y un landmark
estados2 = loc.localizacion_FK(q, r, rango_sensor, dist, t_max, seeds)
estados2_media = sum(estados2)/len(estados2)
plt.figure(2)
plots.plot_estados(estados2_media, "localizacion_FK.pdf", dist)