import matplotlib.pyplot as plt
import localizacion as loc
import plots
import csv
import numpy as np
import pandas as pd
import math


dist = 200    # (m) este parámetro no se cambia
t_max = 5000  # (s) este parámetro no se cambia

#seeds = [123456789, 234567891, 34567891, 456789123, 567891234, 67891234, 7891234, 89123456, 912345678, 213456789]
seeds = [123456789, 123456789]

#q_set = [0.2, 0.4, 0.5, 0.7]
#r_eval = [0.2, 0.3, 0.4, 0.8]
q_set = [0.3]
r_eval = [0.1]
rango_set = [100]

#Ctes iniciales
v_lin = 0.5     #Velocidad lineal m/s
radio = 260

# Landmarks definidos respecto del origen
m = [ (0, 2*radio),
        (0, 0)]

with open('log.csv','w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x_avg', 'y_avg', 'xest_avg', 'yest_avg', 'error_avg', 'x_std', 'y_std', 'xest_std', 'yest_std', 'var_q', 'var_r'])

    for j in enumerate(r_eval):
        r_set = j[1]
        for i in enumerate(q_set):
            q = i[1] ** 2  # prueba con distinto ruido odométrico, varianza (m2)
            r = r_set ** 2  # prueba con distinto ruido sensorial, varianza (m2)
            rango_sensor = rango_set[i[0]]  # prueba con distinto alcance sensorial (m)

            # Localización con FK y landmarks
            estados2 = loc.localizacion2D(q, r, rango_sensor, t_max, m, v_lin, radio, seeds)
            x = []
            y = []
            err = []
            x_est = []
            y_est = []
            zx = []
            zy = []
            Kx = []
            Ky = []
            px = []
            py = []
            for l in range(t_max):
                    x.append(sum(estados2[k]['x'][l][0] for k in range(len(estados2)))/len(estados2))
                    y.append(sum(estados2[k]['x'][l][1] for k in range(len(estados2)))/len(estados2))
                    err.append(sum(estados2[k]['err'][l] for k in range(len(estados2)))/len(estados2))
                    x_est.append(sum(estados2[k]['x_est'][l][0] for k in range(len(estados2)))/len(estados2))
                    y_est.append(sum(estados2[k]['x_est'][l][1] for k in range(len(estados2)))/len(estados2))
                    zx.append(sum(estados2[k]['z'][l][1] for k in range(len(estados2)))/len(estados2))
                    zy.append(sum(estados2[k]['z'][l][1] for k in range(len(estados2)))/len(estados2))
                    Kx.append(sum(estados2[k]['k'][l][0] for k in range(len(estados2)))/len(estados2))
                    Ky.append(sum(estados2[k]['k'][l][1] for k in range(len(estados2)))/len(estados2))
                    px.append(sum(estados2[k]['p'][l][0] for k in range(len(estados2)))/len(estados2))
                    py.append(sum(estados2[k]['p'][l][1] for k in range(len(estados2)))/len(estados2))
            x_mean = np.mean(x)
            x_var = np.std(x)
            y_mean = np.mean(y)
            y_var = np.std(y)
            err_mean = np.mean(err)
            err_var = np.std(err)
            xest_mean = np.mean(x_est)
            yest_mean = np.mean(y_est)
            xest_var = np.std(y_est)
            yest_var = np.std(y_est)
            p = [math.sqrt(px[k]**2 + py[k]**2) for k in range(t_max)]
            K = [math.sqrt(Kx[k]**2 + Ky[k]**2) for k in range(t_max)]
            z = [math.sqrt(zx[k]**2 + zy[k]**2) for k in range(t_max)]
            
            fila = [x_mean, x_var, y_mean, y_var, xest_mean, xest_var, yest_mean, yest_var, err_mean, err_var]
            writer.writerow(fila)
            fila = {'x_est':[x_est], 'x': [x], 'y_est':[y_est], 'y':[y], 'err': [err], 'p': [estados2[0]['p']], 'z': [estados2[0]['z']], 'k': [estados2[0]['k']]}
            estados2_media = pd.DataFrame({'x':x, 'y':y, 'x_est':x_est, 'y_est':y_est, 'err':err, 'z':z, 'k':K, 'p':p })
            plots.plot_estados(estados2_media, i[1], r_set, m, rango_sensor, "media/lfk_q" + str(i[0]) + "r" + str(j[0]) + ".png", dist)


