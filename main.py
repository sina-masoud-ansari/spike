from neuron import *
from network import *
import pylab
import matplotlib.animation as animation

def main():
    network = Network("Initial", 10, init_random_values=True)
    for i in range(10):
        network.update()
        values = network.get_values()



if __name__ == "__main__":
    main()
