# Neural network class
import itertools
from neuron import *
import numpy as np

# Todo: semantics to join networks
# Todo: ability to define distribution for neuron weight_delta and decay rate
# Todo: clamp weights or redifne the way we do inhibitory and excitatory neurons

class Network:
    # size is a tuple of length 3
    def __init__(self, name, shape, density=1.0, ratio_inhibitory=0.5, mean_threshold=0.5, stdev_threshold=0.5, mean_weight=0.1, stdev_weight=0.05, init_random_values=False):

        if not name.strip():
            raise NetworkExcetption("Network name must have a least 1 non-whitespace character")
        else:
            self.name = name

        if len(shape) != 3:
            raise NetworkExcetption("Network size must be a tuple of length 3")
        for i in shape:
            if not isinstance(i, int) or i < 1:
                raise NetworkExcetption("Network size elements must be positive integers")
        else:
            self.shape = shape
            self.size = shape[0] * shape[1] * shape[2]

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

        self.neurons = np.empty(self.size, dtype=object)
        self.ids = itertools.count()

        self.create_neurons()
        self.connect_internal()

        if init_random_values:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    for k in range(self.shape[2]):
                        n = self.neurons[i][j][k]
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
            self.neurons[i] = neuron
        self.neurons = np.reshape(self.neurons, self.shape)

    def connect_internal(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    nu = self.neurons[i][j][k]
                    nu_loc = np.asarray([i, j, k], dtype=np.float)
                    for l in range(self.shape[0]):
                        for m in range(self.shape[1]):
                            for n in range(self.shape[2]):
                                nd = self.neurons[l][m][n]
                                nd_loc = np.asarray([l, m, n], dtype=np.float)
                                distance = np.linalg.norm(nd_loc - nu_loc)
                                if distance > 0:
                                    fd = 1.0 / distance
                                    if np.random.ranf() < self.density * fd:
                                        nu.add_downstream(nd)
                                        weight = abs(self.stdev_weight * np.random.randn() + self.mean_weight)
                                        nd.add_upstream(nu.id, weight)

    def update(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    n = self.neurons[i][j][k]
                    n.integrate()
                    n.fire()

    def get_values(self):
        values = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    n = self.neurons[i][j][k]
                    values[i][j][k] = n.value
        return values

    def get_fired(self):
        fired = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    n = self.neurons[i][j][k]
                    fired[i][j][k] = int(n.fired)
        return fired

    def __repr__(self):
        return "Network: size: {s}, density: {d}, ratio_inhib: {rh}, mean_thresh: {mnt}, stdev_thresh: {sdt}".format(s=self.size, rh=self.ratio_inhibitory, mnt=self.mean_threshold, sdt=self.stdev_threshold)


class NetworkExcetption(Exception):
    pass
