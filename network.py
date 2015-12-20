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
    #OUTPUT_NEURON = 2

    def __init__(self, type, shape, position, spacing, memory_gradient=False, ratio_inhibitory=0.5, threshold=(0.5, 0.5), max_weight_association_delta=(0.01, 0.01), max_weight_decay_delta=(0.04, 0.02), init_random_values=False):

        self.type = type
        self.position = position
        self.spacing = spacing
        self.shape = shape
        self.memory_gradient = memory_gradient
        self.max_weight_association_delta = max_weight_association_delta
        self.max_weight_decay_delta = max_weight_decay_delta

        self.size = shape[0] * shape[1] * shape[2]
        self.network_center = np.asarray(self.position, dtype=float) + 0.5 * (np.asarray(self.shape, dtype=float) - 1)* np.asarray(self.spacing, dtype=float)

        """
        np_position = np.asarray(self.position)
        np_shape = np.asarray(self.shape)
        np_spacing = np.asarray(self.spacing)
        np_distance = np.linalg.norm(np_shape * np_spacing)
        self.max_distance = np.linalg.norm(np_distance)
        """

        if ratio_inhibitory < 0:
            raise NetworkExcetption("Network ratio_inhibitory must be >= 0")
        else:
            self.ratio_inhibitory = ratio_inhibitory

        if not threshold[0] > 0 and threshold[1] > 0:
            raise NetworkExcetption("Network threshold mean and stdev must be > 0")
        else:
            self.threshold = threshold

        self.neurons = np.empty(self.shape, dtype=object)
        self.create_neurons()

        if init_random_values:
            self.random_activation()

    # exponential probability density function
    def exp_pdf(self, x):
        if x > 0:
            return np.exp(-x)
        else :
            return 0

    def linear_normalised_pdf(self, x, d):
        if x < 0 or x > d:
            return 0
        else:
            return 1.0 - x/d

    def random_activation(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    n = self.neurons[i][j][k]
                    n.value = np.random.ranf()


    def create_neurons(self):
        x, y, z = self.position
        dx, dy, dz = self.spacing
        mean_threshold, stdev_threshold = self.threshold
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    if np.random.ranf() < self.ratio_inhibitory:
                        type = Neuron.I
                    else:
                        type = Neuron.E
                    threshold = abs(mean_threshold * np.random.randn() + stdev_threshold)
                    position = (x + i * dx, y + j * dy, z + k * dz)
                    if self.type == self.STDP_NEURON:
                        p = 1.0
                        if self.memory_gradient:
                            distance = np.linalg.norm(self.network_center - position)
                            #p = 1.0 - self.exp_pdf(distance) + 1e-6
                            p = 1.0 - 1.0 / (distance * distance + 1e-6)
                            p = min(1.0, p)
                        # weight association - outer nodes are more adaptive
                        wad_mean, wad_stdev = self.max_weight_association_delta
                        wad = (wad_mean * p, wad_stdev * p)
                        # weight decay - inner nodes preserve patterns for longer
                        wdd_mean, wdd_stdev = self.max_weight_decay_delta
                        wdd = (wdd_mean * p, wdd_stdev * p)
                        self.neurons[i][j][k] = Neuron(type, threshold, position, weight_association_delta=wad, weight_decay_delta=wdd)
                    elif self.type == self.SENSOR_NEURON:
                        self.neurons[i][j][k] = Sensor(threshold, position)
                    #elif self.type == self.OUTPUT_NEURON:
                    #    self.neurons[i][j][k] = Output(type, threshold, position)

    def connect(self, other, density=0.05, connection_weights=(0.1, 0.05)):
        if not isinstance(density, float) or not 0.0 < density <= 1.0:
            raise NetworkExcetption("Network connection density must a positive float <= 1.0 ")

        if not connection_weights[0] > 0 and not connection_weights[1] > 0:
            raise NetworkExcetption("Network connection mean weight and stdev must be > 0")

        mean_connection_weight, stdev_connection_weight = connection_weights
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
                                #p = self.exp_pdf(distance) * density
                                p = density / (distance * distance + 1e-3)
                                p = min(1.0, p)
                                if np.random.ranf() < p:
                                    nu.add_downstream(nd)
                                    weight = abs(stdev_connection_weight * np.random.randn() + mean_connection_weight)
                                    nd.add_upstream(nu, weight)


    def update(self):
        # fire all potential neurons
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    n = self.neurons[i][j][k]
                    n.integrate_and_fire()

        if self.type in [self.STDP_NEURON]:
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
                    values[i][j][k] = n.value / n.threshold
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

