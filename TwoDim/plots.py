import matplotlib.pyplot as plt

def plot_estados(estados, q, r, file=None, ymax=None):
    fig, axes = plt.subplots(5, 1)

    axes[0].plot(estados['x'], linewidth=0.8)
    axes[0].plot(estados['xest'], linewidth=0.8)
    axes[0].set_ylabel('x (azul)\nx_est (rojo)')
    plt.suptitle("q = " + str(q) + ", r = " + str(r), fontsize = 14 )
    if (ymax is not None):
        axes[0].set_ylim(-ymax * 0.05, ymax * 1.05)

    axes[1].set_ylabel('xerror')
    axes[1].plot(estados['xerror'], linewidth=0.8)

    axes[2].set_ylabel('Z')
    axes[2].plot(estados['z'], linewidth=0.8)
    axes[2].set_ylim(-0.9)

    axes[3].set_ylabel('K')
    axes[3].plot(estados['K'], linewidth=0.8)
    axes[3].set_ylim(0, 1)

    axes[4].set_ylabel('P')
    axes[4].plot(estados['p'], linewidth=0.8)

    for i in range(5):
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].grid(linestyle=':', linewidth=0.3)
        if (i < 4):
            axes[i].set_xticklabels([])

    if (file is None):
        plt.show()
    else:
        plt.savefig(file)
    plt.close()

    


