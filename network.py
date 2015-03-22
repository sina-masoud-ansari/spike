# Neural network class
import itertools
from neuron import *
import numpy as np

# Todo: semantics to join networks
# Todo: ability to define distribution for neuron weight_delta and decay rate

class Network:
    def __init__(self, name, size, density, ratio_inhibitory=0.3, min_threshold=0.1, max_threshold=100):

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

        if not min_threshold > 0:
            raise NetworkExcetption("Network min_threshold must be > 0")
        else:
            self.min_threshold = min_threshold

        if not max_threshold > self.min_threshold:
            raise NetworkExcetption("Network max_threshold must be > min_threshold")
        else:
            self.max_threshold = max_threshold

        self.neurons = []
        self.ids = itertools.count

    def create_neurons(self):
        for i in range(self.size):
            id = next(self.ids)
            if np.random.rand() < self.ratio_inhibitory:
                type = Neuron.I
            else:
                type = Neuron.E

            threshold = max(self.min_threshold, np.random.rand() * self.max_threshold)

            neuron = Neuron(self.name, id, type, threshold)
            self.neurons.append(neuron)

    def connect_internal(self):
        pass

    def __repr__(self):
        return "Network: size: {s}, density: {d}, ratio_inhib: {rh}, min_thresh: {mnt}, max_thresh: {mxt}".format(s=self.size, rh=self.ratio_inhibitory, mnt=self.min_threshold, mxt=self.max_threshold)


class NetworkExcetption(Exception):
    pass
