import math
import numpy as np
from numpy.linalg import inv
import random as rnd
import pandas as pd
import matplotlib.pyplot as plt

q = 0.2 ** 2
r = 0.2 ** 2
rango_sensor = 5
t_max = 5000

estados = []

#Constantes iniciales
v_lin = 0.5     #Velocidad lineal m/s
radio = 15
v_ang = v_lin / radio     #Velocidad angular rad/s
x = np.array([[0], [0]], dtype=float)    #Posición inicial 0,0
ang = 0
l = math.sin(v_ang)
k = math.cos(v_lin)

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
m = [ [[0],[0]], 
      [[20],[40]],
      [[80],[100]]]

#estados.append(pd.DataFrame(data={'x': [], 'xest': [], 'xerror': [], 'p': [], 'z': [], 'K': [], 'q': [], 'r': []}))
estados = pd.DataFrame(estados.append(pd.DataFrame(data = {'x':[]})))

#Simulación
for t in range(t_max):
    #Calculo los números aleatorios
    r1 = rnd.gauss(0,1)
    r2 = rnd.gauss(0,1)

    # Para dar más realismo, se provoca un ruido de velocidad asimétrico;
    # en este caso se hace más probable que la velocidad sea mayor a la esperada
    if (rnd.random() > 0.7):
        r1 = math.fabs(r1) * (-1 if v_lin<0 else 1)

    #Posiciones en los ejes respecto a las anteriores
    ang += v_ang
    x[0] += v_lin * math.cos(ang)
    x[1] += v_lin * math.sin(ang)

    fila = {'x': x[0], 'y':x[1]}
    estados = pd.concat([estados, pd.DataFrame(fila)], ignore_index=True )

plt.figure(num="MovimientoCircular",figsize=(4,4))
plt.plot(estados['x'],estados['y'])
#plt.axis([-0.5,0.5,-0.5,0.5])
plt.show()
