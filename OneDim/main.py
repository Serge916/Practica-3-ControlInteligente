import matplotlib.pyplot as plt
import localizacion as loc
import plots
import csv
import MergeImages as merge


dist = 200    # (m) este parámetro no se cambia
t_max = 5000  # (s) este parámetro no se cambia

#seeds = [123456789, 234567891, 34567891, 456789123, 567891234, 67891234, 7891234, 89123456, 912345678, 213456789]
seeds = [123456789]

#q_set = [0.2, 0.4, 0.5, 0.7]
#r_eval = [0.2, 0.3, 0.4, 0.8]
q_set = [0.8]
r_eval = [0.7]
rango_set = [5]

with open('log.csv','w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'xest', 'xerror', 'p', 'z', 'K', 'var_q', 'var_r'])

    for j in enumerate(r_eval):
        r_set = j[1]
        for i in enumerate(q_set):
            q = i[1] ** 2  # prueba con distinto ruido odométrico, varianza (m2)
            r = r_set ** 2  # prueba con distinto ruido sensorial, varianza (m2)
            #rango_sensor = rango_set[i[0]]  # prueba con distinto alcance sensorial (m)
            rango_sensor = 5

            # Localización por odometría
            estados1 = loc.localizacion_odometrica(q, r, rango_sensor, dist, t_max, seeds[0])
            plots.plot_estados(estados1, i[1], r_set, "media/lo_q" + str(i[0]) + "r" + str(j[0]) + ".png", dist)
            writer.writerow(estados1.mean())

            # Localización con FK y un landmark
            estados2 = loc.localizacion_FK(q, r, rango_sensor, dist, t_max, seeds)
            writer.writerow(estados2[0].mean())
            estados2_media = sum(estados2)/len(estados2)
            plt.figure(2)
            plots.plot_estados(estados2_media, i[1], r_set, "media/lfk_q" + str(i[0]) + "r" + str(j[0]) + ".png", dist)

merge.make_image("media/lfk*.png").save("Merged_LFK.png")
merge.make_image("media/lo*.png").save("Merged_LO.png")
