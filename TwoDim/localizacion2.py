import math
import numpy as np
from numpy.linalg import inv
import random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

q = 0.05 ** 2
r = 0.2 ** 2
rango_sensor = 50
t_max = 5000

estados = []

#Constantes iniciales
v_lin = 0.5     #Velocidad lineal m/s
radio = 150
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
m = [ (0, 0), 
      (0, 300),
      (150, 150),
      (-150, 150)]

#estados.append(pd.DataFrame(data={'x': [], 'xest': [], 'xerror': [], 'p': [], 'z': [], 'K': [], 'q': [], 'r': []}))
estados = pd.DataFrame(estados.append(pd.DataFrame(data = {'x_est':[], 'x': []})))

#Simulación
for t in range(t_max):
    #Calculo los números aleatorios
    r1 = rnd.gauss(0,1)
    r2 = rnd.gauss(0,1)

    #Posiciones ideales
    ang_id += v_ang
    x_est[0][0] += v_lin * math.cos(ang_id)
    x_est[1][0] += v_lin * math.sin(ang_id)

    #Posiciones reales
    ang += v_ang + r1/(20*math.pi) * (q**0.5)
    x[0][0] += v_lin * math.cos(ang) + r1 * (q**0.5)
    x[1][0] += v_lin * math.sin(ang) + r1 * (q**0.5)
  
    #Filtro
    P = P + Q  # varianza del error asociada a la estimación a priori
    correccion = False

    z = []
    for mi in m:
        if (math.fabs( math.sqrt((x[0][0])**2 + (x[1][0])**2) - mi[0]) < rango_sensor):  # si se detecta el landmark
            # Actualizamos la observación (con ruido)
            foo = np.array([[x[0][0] + r2 * (r**0.5)], [x[1][0] + r2 * (r**0.5)]])
            correccion = True
        else:
            foo = np.array([[math.nan], [math.nan]])
        z.append(foo)

    # Corrección basada en z0, sólo si se detectó algún landmark
    if (correccion):
        # Actualización de la medición (innovación)
        innov = points = 0
        for zi in z:
            if not math.isnan(zi[0]):
                innov += zi - H * x_est
                points += 1
        innov /= points         #Hago la media

        # Ganancia de Kalman
        K = P * np.transpose(H) * inv(H * P * np.transpose(H) + R)

        # Estimación a posteriori
        x_est = x_est + K * innov

        # Covarianza del error asociada a la estimación a posteriori
        P = P - K * H * P
    else:
        K = np.array([[math.nan, 0],
                        [ 0, 0]])  # 2x2 para (x,y)

    fila = {'x_est': [(x_est[0][0], x_est[1][0])], 'x': [(x[0][0], x[1][0])]}
    estados = pd.concat([estados, pd.DataFrame(fila)], ignore_index=True )


disp = plt.figure(num="MovimientoCircular",figsize=(5,5))
c,v = zip(*estados['x_est'])
plt.plot(c, v, linewidth=1 , linestyle='dashed' , label="Estimado")
c,v = zip(*estados['x'])
plt.plot(c, v, linewidth=1 , label="Real")

angles = np.linspace(0, 2*math.pi, 150)
a = np.array([rango_sensor * math.cos(angle) for angle in angles])
b = np.array([rango_sensor * math.sin(angle) for angle in angles])


for mi in m:
    plt.plot(a + mi[0], b + mi[1], linestyle='dashed', color='green', linewidth=1)
    plt.plot(mi[0], mi[1], "x", color='green', linewidth=1)
plt.plot(0,0, "x", color='green', label="Landmark")

plt.legend(loc='upper left')
plt.show()
