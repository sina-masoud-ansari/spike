# Neural network class

from neuron import *
from sensor import *
from output import *
import numpy as np

# Todo: semantics to join networks
# Todo: ability to define distribution for neuron weight_delta and decay rate
# Todo: clamp weights or redifne the way we do inhibitory and excitatory neurons

class Network:

    STDP_NEURON = 0
    SENSOR_NEURON = 1
    OUTPUT_NEURON = 2

    def __init__(self, type, shape, position, spacing, ratio_inhibitory=0.5, mean_threshold=0.5, stdev_threshold=0.5, mean_weight=0.1, stdev_weight=0.05, init_random_values=False):

        self.type = type
        self.position = position
        self.spacing = spacing
        self.shape = shape
        self.size = shape[0] * shape[1] * shape[2]

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

        self.neurons = np.empty(self.shape, dtype=object)
        self.create_neurons()

        if init_random_values:
            self.random_activation()

    def random_activation(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    n = self.neurons[i][j][k]
                    n.value = np.random.ranf()


    def create_neurons(self):
        dx, dy, dz = self.spacing
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    if np.random.ranf() < self.ratio_inhibitory:
                        type = Neuron.I
                    else:
                        type = Neuron.E
                    threshold = abs(self.stdev_threshold * np.random.randn() + self.mean_threshold)
                    x, y, z = self.position
                    position = (x + i * dx, y + j * dy, z + k * dz)
                    if self.type == self.STDP_NEURON:
                        self.neurons[i][j][k] = Neuron(type, threshold, position)
                    elif self.type == self.SENSOR_NEURON:
                        self.neurons[i][j][k] = Sensor(threshold, position)
                    elif self.type == self.OUTPUT_NEURON:
                        self.neurons[i][j][k] = Output(type, threshold, position)

    def connect(self, other, density=0.05):
        if not isinstance(density, float) or not 0.0 < density <= 1.0:
            raise NetworkExcetption("Network connection density must a positive float <= 1.0 ")

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    nu = self.neurons[i][j][k]
                    nu_loc = np.asarray(nu.position, dtype=np.float)

                    for l in range(other.shape[0]):
                        for m in range(other.shape[1]):
                            for n in range(other.shape[2]):
                                nd = other.neurons[l][m][n]
                                nd_loc = np.asarray(nd.position, dtype=np.float)
                                distance = np.linalg.norm(nd_loc - nu_loc)
                                if distance > 0:
                                    fd = 1.0 / distance
                                    if np.random.ranf() < density * fd:
                                        nu.add_downstream(nd)
                                        weight = abs(self.stdev_weight * np.random.randn() + self.mean_weight)
                                        nd.add_upstream(nu, weight)

    def update(self):
        # fire all potential neurons
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    n = self.neurons[i][j][k]
                    n.integrate_and_fire()

        if self.type in [self.STDP_NEURON, self.OUTPUT_NEURON]:
            # update the weights
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    for k in range(self.shape[2]):
                        n = self.neurons[i][j][k]
                        n.update_weights()


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

    def apply_random_input(self, values):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    self.neurons[i][j][k].activate(values[i][j][k])

    def __repr__(self):
        return "Network: size: {s}, density: {d}, ratio_inhib: {rh}, mean_thresh: {mnt}, stdev_thresh: {sdt}".format(s=self.size, rh=self.ratio_inhibitory, mnt=self.mean_threshold, sdt=self.stdev_threshold)



class NetworkExcetption(Exception):
    pass

