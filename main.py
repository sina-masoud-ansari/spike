import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
#import numpy as np
from network import *


def update(frame, points, network):
    network.update()
    values = network.get_values()
    points.set_array(values.flatten())


def main():
    shape = (5, 10, 5)
    network = Network("Initial", shape, density=0.3, init_random_values=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = []
    ys = []
    zs = []

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                xs.append(x)
                ys.append(y)
                zs.append(z)

    values = network.get_values()
    points = ax.scatter(xs, ys, zs, s=40, depthshade=True, cmap=cm.jet, c=values.flatten())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ani = animation.FuncAnimation(fig, update, fargs=(points, network), interval=0)
    plt.show()

if __name__ == "__main__":
    main()
