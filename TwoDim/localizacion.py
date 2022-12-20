import numpy as np
import pandas as pd
import math
from numpy.linalg import inv
import random as rnd


def localizacion2D(q, r, rango_sensor, t_max, m, v_lin, radio, seeds):
    estados = []
    for i,s in enumerate(seeds):
        rnd.seed(s)
    
        #Constantes iniciales
        v_ang = v_lin / radio     #Velocidad angular rad/s
        x = np.array([[0], [0]], dtype=float)    #Posición inicial 0,0
        ang = ang_id = 0

        #Movimiento
        x_est = x.copy()    #Empiezo en el origen


        # Covarianza del error asociada a la estimación a priori
        P = np.array([[0, 0],
                    [0, 0]])  # 2x2 para (x,y)
        # Ruido del modelo de movimiento
        Q = np.array([[q, 0],
                    [0, q]])  # 2x2 para (x,y)
        # Ruido sensorial
        R = np.array([[r, 0],
                    [0, r]])  # 2x2 para (x,y)

        # Relación entre mediciones y vector de estado
        H = np.array([[1, 0],
                    [0, 1]])  # 2x2 para (x,y)

        # Valor inicial de K, solo para incluirlo en 'estados' antes de calcularlo por primera vez
        K = np.array([[0, 0],
                    [0, 0]])  # 2x2 para (x,y)

        estados.append(pd.DataFrame(data = {'x_est':[], 'x': [], 'err': [], 'p': [], 'z': [], 'k': []}))

        #Simulación
        for t in range(t_max):
            #Calculo los números aleatorios
            r1 = rnd.gauss(0,1)
            r2 = rnd.gauss(0,1)
            r3 = rnd.gauss(0,1)
            r4 = rnd.gauss(0,1)
            r5 = rnd.gauss(0,1)

            #Posiciones ideales
            ang_id += v_ang
            x_est[0][0] += v_lin * math.cos(ang_id)
            x_est[1][0] += v_lin * math.sin(ang_id)

            #Posiciones reales
            ang += v_ang + r1/(20*math.pi) * (q**0.5)
            x[0][0] += (v_lin + r2 * (q**0.5)) * math.cos(ang) 
            x[1][0] += (v_lin + r3 * (q**0.5)) * math.sin(ang)
        
            #Filtro
            P = P + Q  # varianza del error asociada a la estimación a priori
            correccion = False

            z = []
            for mi in m:
                if ( math.fabs(( math.sqrt( (mi[0] - x[0][0])**2 + (mi[1] - x[1][0])**2) ) ) < rango_sensor):  # si se detecta el landmark
                    # Actualizamos la observación (con ruido)
                    foo = np.array([[x[0][0] + r4 * (r**0.5)], [x[1][0] + r5 * (r**0.5)]])
                    correccion = True
                else:   
                    foo = np.array([[math.nan], [math.nan]])
                z.append(foo)

            # Corrección basada en z, sólo si se detectó algún landmark
            if (correccion):
                # Actualización de la medición (innovación)
                points = 0
                innov = np.array([[0], [0]], dtype=float)
                for zi in z:
                    if not math.isnan(zi[0]):
                        innov = innov + zi - H @ x_est
                        points += 1
                innov = innov / points         #Hago la media

                # Ganancia de Kalman
                K = P @ np.transpose(H) @ inv(H @ P @ np.transpose(H) + R)

                # Estimación a posteriori
                x_est = x_est + K @ innov

                # Covarianza del error asociada a la estimación a posteriori
                P = P - K * H * P
            else:
                K = np.array([[math.nan, 0],
                                [ 0, math.nan]])  # 2x2 para (x,y)

            fila = {'x_est': [(x_est[0][0], x_est[1][0])], 'x': [(x[0][0], x[1][0])], 'err':[math.fabs(x[1][0]-x_est[1][0]+x[0][0]-x_est[0][0])],
                    'p': [(P[0][0], P[1][1])], 'k': [(K[0][0], K[1][1])], 'z': [(z[0][0], z[1][0])]}
            estados[i] = pd.concat([estados[i], pd.DataFrame(fila)], ignore_index=True )
    return estados
