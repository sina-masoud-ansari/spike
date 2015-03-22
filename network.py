# Neural network class
import itertools
from neuron import *
import numpy as np

# Todo: semantics to join networks
# Todo: ability to define distribution for neuron weight_delta and decay rate
# Todo: add structure (1D, 2D, 3D)
# Todo: localised connections i.e probability based on distance

class Network:
    def __init__(self, name, size, density=0.3, ratio_inhibitory=0.3, mean_threshold=0.5, stdev_threshold=0.5, mean_weight=0.1, stdev_weight=0.1, init_random_values=False):

        if not name.strip():
            raise NetworkExcetption("Network name must have a least 1 non-whitespace character")
        else:
            self.name = name

        if not isinstance(size, int) or size < 1:
            raise NetworkExcetption("Network size must be a positive integer")
        else:
            self.size = size

        if not isinstance(density, float) or not 0.0 < density <= 1.0:
            raise NetworkExcetption("Network density must a positive float <= 1.0 ")
        else:
            self.density = density

        if ratio_inhibitory < 0:
            raise NetworkExcetption("Network ratio_inhibitory must be >= 0")
        else:
            self.ratio_inhibitory = ratio_inhibitory

        if not mean_threshold > 0:
            raise NetworkExcetption("Network mean_threshold must be > 0")
        else:
            self.mean_threshold = mean_threshold

        if stdev_threshold < 0:
            raise NetworkExcetption("Network stdev_threshold must be >= 0")
        else:
            self.stdev_threshold = stdev_threshold

        if not mean_weight > 0:
            raise NetworkExcetption("Network mean_weight must be > 0")
        else:
            self.mean_weight = mean_weight

        if stdev_weight < 0:
            raise NetworkExcetption("Network stdev_weight must be >= 0")
        else:
            self.stdev_weight = stdev_weight

        self.neurons = []
        self.ids = itertools.count()

        self.create_neurons()
        self.connect_internal()

        if init_random_values:
            for n in self.neurons:
                n.value = np.random.ranf()

    def create_neurons(self):
        for i in range(self.size):
            id = next(self.ids)
            if np.random.ranf() < self.ratio_inhibitory:
                type = Neuron.I
            else:
                type = Neuron.E

            threshold = abs(self.stdev_threshold * np.random.randn() + self.mean_threshold)
            neuron = Neuron(self.name, id, type, threshold)
            self.neurons.append(neuron)

    def connect_internal(self):
        for nu in self.neurons:
            for nd in self.neurons:
                if nu != nd and np.random.ranf() < self.density:
                    nu.add_downstream(nd)
                    weight = abs(self.stdev_weight * np.random.randn() + self.mean_weight)
                    nd.add_upstream(nu.id, weight)

    def update(self):
        for n in self.neurons:
            n.integrate()
            n.fire()

    def get_values(self):
        values = np.zeros(self.size)
        for i, n in enumerate(self.neurons):
            values[i] = n.value
        return values

    def __repr__(self):
        return "Network: size: {s}, density: {d}, ratio_inhib: {rh}, mean_thresh: {mnt}, stdev_thresh: {sdt}".format(s=self.size, rh=self.ratio_inhibitory, mnt=self.mean_threshold, sdt=self.stdev_threshold)


class NetworkExcetption(Exception):
    pass
