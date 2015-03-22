from neuron import *
from network import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

size = 35
network = None
cmap = 'binary'
fig = plt.figure()
im = plt.imshow(np.zeros((size, size)), cmap=cmap, interpolation='none')
plt.tick_params(axis='both',bottom='off',left='off', labelbottom='off',labelleft='off')



def update(*args):
    global network
    network.update()
    #values = network.get_values().reshape((size, size))
    fired = network.get_fired().reshape((size, size))
    im.set_data(fired)
    im.autoscale()

    return im


def main():
    global network
    network = Network("Initial", size*size, density=0.3, init_random_values=True)
    ani = animation.FuncAnimation(fig, update, interval=250)
    plt.show()



if __name__ == "__main__":
    main()
