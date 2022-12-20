import math
import numpy as np
from numpy.linalg import inv
import random as rnd
import pandas as pd

def localizacion_odometrica(q, r, rango_sensor, dist, t_max, seed):
    '''
    :param q: varianza de velocidad (metros^2)
    :param r: varianza del sensor (metros^2)
    :param rango_sensor: alcance máximo del sensor (metros)
    :param dist: distancia del pasillo (metros)
    :param t_max: tiempo de simulación (segundos)
    :param seed: semilla para generar números aleatorios
    :return: estados (dataframe) con los valores en cada t
    '''

    # Inicializa la secuencia pseudo-aleatoria
    rnd.seed(seed)

    # Condiciones iniciales
    v = 0.5  # velocidad lineal constante 50 cm/s
    v_est = 0.5  # velocidad lineal constante 50 cm/s
    x = 0  # posicion inicial
    direccion = 1  # 1: hacia la derecha, -1: hacia la izquierda
    direccion_est = 1  # 1: hacia la derecha, -1: hacia la izquierda

    x_est = 0  # Posición en x
    p = 0  # Varianza en x

    estados = pd.DataFrame(data={'x': [], 'xest': [], 'xerror': [], 'p': [], 'z': [], 'K': [], 'q': [], 'r': []})

    # Simulación
    for t in range(t_max):
        # Se calculan todos los números aleatorios que se puedan
        # usar para mantener la misma secuencia; útil para comparar
        r1 = rnd.gauss(0, 1)
        r2 = rnd.gauss(0, 1)

        # Para dar más realismo, se provoca un ruido de velocidad asimétrico;
        # en este caso se hace más probable que la velocidad sea mayor a la esperada
        if (rnd.random() > 0.7):
            r1 = math.fabs(r1) * (-1 if v < 0 else 1)

        # Dirección
        if (x > dist and direccion == 1):
            direccion = -1
            v = -v
        elif (x < 0 and direccion == -1):
            direccion = 1
            v = -v

        # Dirección
        if (x_est > dist and direccion_est == 1):
            direccion_est = -1
            v_est = -v_est
        elif (x_est < 0 and direccion_est == -1):
            direccion_est = 1
            v_est = -v_est

        # Nuevo estado real
        x = x + v + r1 * q ** .5

        # Estimación a priori
        x_est = x_est + v_est  # Estimacion del nuevo estado
        p = p + q  # Varianza del error asociada a la estimacion a priori

        fila = {'x': [x], 'xest': [x_est], 'xerror': [math.fabs(x - x_est)], 'p': [p], 'z': [math.nan], 'K': [math.nan], 'q': [q], 'r': [r]}
        estados = pd.concat([estados, pd.DataFrame(fila)], ignore_index=True)
    return estados


def localizacion_FK(q, r, rango_sensor, dist, t_max, seeds):
    '''
    :param q: varianza de velocidad (metros^2)
    :param r: varianza del sensor (metros^2)
    :param rango_sensor: alcance máximo del sensor (metros)
    :param dist: distancia del pasillo (metros)
    :param t_max: tiempo de simulación (segundos)
    :param seeds: lista de semillas para generar números aleatorios en varias ejecuciones
    :return: estados (dataframe) con los valores en cada t
    '''

    estados = []

    for i,s in enumerate(seeds):
        # Inicializa la secuencia pseudo-aleatoria
        rnd.seed(s)

        # Condiciones iniciales
        v = 0.5  # velocidad lineal constante 50 cm/s
        v_est = 0.5  # velocidad lineal constante 50 cm/s
        x = np.array([[0], [0]])  # posición inicial
        direccion = 1  # 1: hacia la derecha, -1: hacia la izquierda
        direccion_est = 1  # 1: hacia la derecha, -1: hacia la izquierda

        # Vector de estado (x,y)
        #    x_est = np.transpose(np.array([x, 0]))  # 1x2 para (x,y)
        x_est = x.copy()  # 1x2 para (x,y)

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
        m = [0, 100, 200]

        estados.append(pd.DataFrame(data={'x': [], 'xest': [], 'xerror': [], 'p': [], 'z': [], 'K': [], 'q': [], 'r': []}))

        # Simulación
        for t in range(t_max):
            # Se calculan todos los números aleatorios que se puedan
            # usar para mantener la misma secuencia; útil para comparar
            r1 = rnd.gauss(0, 1)
            r2 = rnd.gauss(0, 1)

            # Para dar más realismo, se provoca un ruido de velocidad asimétrico;
            # en este caso se hace más probable que la velocidad sea mayor a la esperada
            if (rnd.random() > 0.7):
                r1 = math.fabs(r1) * (-1 if v<0 else 1)

            # Dirección
            if (x[0][0] > dist and direccion == 1):
                direccion = -1
                v = -v
            elif (x[0][0] < 0 and direccion == -1):
                direccion = 1
                v = -v

            # Dirección
            if (x_est[0][0] > dist and direccion_est == 1):
                direccion_est = -1
                v_est = -v_est
            elif (x_est[0][0] < 0 and direccion_est == -1):
                direccion_est = 1
                v_est = -v_est

            # Nuevo estado real
            x = x + v + r1 * (q**0.5)

            # Estimación a priori
            x_est = x_est + np.transpose(np.array([v_est, 0]))  # estimación del nuevo estado

            P = P + Q  # varianza del error asociada a la estimación a priori

            correccion = False

            z = []
            for mi in m:
                if (math.fabs(x[0][0] - mi) < rango_sensor):  # si se detecta el landmark
                    # Actualizamos la observación (con ruido)
                    foo = np.array([[x[0][0] + r2 * (r**0.5)], [0]])
                    correccion = True
                else:
                    foo = np.array([[math.nan], [0]])
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

            fila = {'x': [x[0][0]], 'xest': [x_est[0][0]], 'xerror': [math.fabs(x[0][0] - x_est[0][0])], 'p': [P[0][0]],
                    'z': [z[0][0]], 'K': [K[0][0]], 'q': [q], 'r': [r]}
            estados[i] = pd.concat([estados[i], pd.DataFrame(fila)], ignore_index=True)
                   
    return estados