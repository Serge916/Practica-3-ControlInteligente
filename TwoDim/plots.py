import matplotlib.pyplot as plt
import math
import numpy as np

def plot_estados(estados, q, r, m, rango_sensor, file=None, ymax=None):
   # '''
    disp = plt.figure(num="MovimientoCircular",figsize=(5,5))
    plt.plot(estados['x_est'],estados['y_est'], linestyle='dashed', linewidth=1 , label="Estimado")
    plt.plot(estados['x'], estados['y'], linewidth=1, label="Real")

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
    #'''
    fig, axes = plt.subplots(6,1)
    timesim = list(range(0, len(estados)))
    axes[0].plot(timesim, estados['x'], linewidth=0.8)
    axes[0].plot(timesim, estados['x_est'], linewidth=0.8)
    axes[0].set_ylabel('x (azul)\ x_est (rojo)')
    plt.suptitle("q = " + str(q) + ", r = " + str(r), fontsize = 14 )

    axes[1].plot(timesim, estados['y'], linewidth=0.8)
    axes[1].plot(timesim, estados['y_est'], linewidth=0.8)
    axes[1].set_ylabel('y (azul)\ y_est (rojo)')

    axes[2].set_ylabel('err')
    axes[2].plot(estados['err'], linewidth=0.8)

    axes[3].set_ylabel('Z')
    axes[3].plot(timesim, estados['z'], linewidth=0.8)

    axes[4].set_ylabel('K')
    axes[4].plot(timesim, estados['k'], linewidth=0.8)


    axes[5].set_ylabel('P')
    axes[5].plot(timesim, estados['p'], linewidth=0.8)

    for i in range(6):
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].grid(linestyle=':', linewidth=0.3)
        if (i < 5):
            axes[i].set_xticklabels([])

    if (file is None):
        plt.show()
    else:
        plt.savefig(file)
    plt.close()

    


