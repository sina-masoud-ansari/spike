from neuron import *
from network import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm


shape = (14, 14, 1)
network = None
cmap = 'binary'
cmap = 'hot'
fig = plt.figure()
im = plt.imshow(np.zeros((shape[0], shape[1])), cmap=cmap, interpolation='none')
plt.tick_params(axis='both',bottom='off',left='off', labelbottom='off',labelleft='off')



def update(*args):
    global network
    network.update()
    values = network.get_values().reshape((shape[0], shape[1]))
    fired = network.get_fired().reshape((shape[0], shape[1]))
    im.set_data(fired)
    im.set_data(values)
    im.autoscale()

    return im


def main():
    global network
    network = Network("Initial", shape, density=1.0, init_random_values=True)
    ani = animation.FuncAnimation(fig, update, interval=10)
    plt.show()

if __name__ == "__main__":
    main()
