import math
import numpy as np
from numpy.linalg import inv
import random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

q = 0.3 ** 2
r = 0.1 ** 2
rango_sensor = 100
t_max = 5000


#Constantes iniciales
v_lin = 0.5     #Velocidad lineal m/s
radio = 260
v_ang = v_lin / radio     #Velocidad angular rad/s
x = np.array([[0], [0]], dtype=float)    #Posición inicial 0,0
ang = ang_id = ang_est= 0

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

# Landmarks definidos respecto del origen
m = [ (0, 2*radio),
        (0, 0)]

estados = pd.DataFrame(data = {'x_est':[], 'x': [], 'err': [], 'p': [], 'z': [], 'k': []})



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
    estados = pd.concat([estados, pd.DataFrame(fila)], ignore_index=True )

disp = plt.figure(num="MovimientoCircular",figsize=(5,5))
c,v = zip(*estados['x_est'])
plt.plot(c, v, linewidth=1 , linestyle='dashed' , label="Estimado")
c,v = zip(*estados['x'])
plt.plot(c, v, linewidth=1 , label="Real")

#Circunferencia de radio = rango_sensor
angles = np.linspace(0, 2*math.pi, 150)
a = np.array([rango_sensor * math.cos(angle) for angle in angles])
b = np.array([rango_sensor * math.sin(angle) for angle in angles])

#Muevo la circunferencia a cada uno de los landmarks
for mi in m:
    ai = a + mi[0]
    bi = b + mi[1]
    plt.plot(ai, bi, linestyle='dashed', color='green', linewidth=1)
    plt.plot(mi[0], mi[1], "x", color='green', linewidth=1)
plt.plot(m[0][0], m[0][1], "x", color='green', label="Landmark")

plt.legend(loc='upper left')
plt.title("Parte 2")
plt.show()

plt.close()

fig, axes = plt.subplots(6,1)
timesim = list(range(0,t_max))
c1,v1 = zip(*estados['x'])
c,v = zip(*estados['x_est'])
axes[0].plot(timesim, c1, linewidth=0.8)
axes[0].plot(timesim, c, linewidth=0.8)
axes[0].set_ylabel('x (azul)\ x_est (rojo)')

axes[1].plot(timesim, v1, linewidth=0.8)
axes[1].plot(timesim, v, linewidth=0.8)
axes[1].set_ylabel('y (azul)\ y_est (rojo)')

axes[2].set_ylabel('err')
axes[2].plot(estados['err'], linewidth=0.8)

axes[3].set_ylabel('Z')
c,v = zip(*estados['z'])
axes[3].plot(timesim, c, linewidth=0.8)
axes[3].plot(timesim, v, linewidth=0.8)

axes[4].set_ylabel('K')
c,v = zip(*estados['k'])
axes[4].plot(timesim, c, linewidth=0.8)
axes[4].plot(timesim, v, linewidth=0.8)


axes[5].set_ylabel('P')
c,v = zip(*estados['p'])
axes[5].plot(timesim, c, linewidth=0.8)
axes[5].plot(timesim, v, linewidth=0.8)

for i in range(6):
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].grid(linestyle=':', linewidth=0.3)
    if (i < 5):
        axes[i].set_xticklabels([])

plt.show()
