# Class for spiking neuron
import itertools
from util import *

class Neuron:
    """
    Discrete spiking neuron
    Neuron will spike when neuron value reaches neuron threshold
    Spike value is determined by neuron type (1 for excitatory and -1 for inhibitory)
    Threshold is a positive float
    Thresholds should be allocated externally according to something like a Gaussian with a mean and stdev defined as
    network parameters.
    Neurons require an ID (non-negative integer)
    They are initialised with 0 value
    Upstream neurons are those that excite or inhibit the neuron
    Downstream neurons are excited or inhibited by the neuron
    """

    # Neuron Type
    I = -1 # Inhibitory
    E = 1 # Excitatory
    MIN_THRESHOLD = 0.001

    def __init__(self, network_name, id, type, threshold, weight_delta=0.01, decay_rate=0.01):

        if not network_name.strip():
            raise NeuronException("Neuron network_name must have a least 1 non-whitespace character")
        else:
            self.network_name = network_name

        if not isinstance(id, int) or id < 0:
            raise NeuronException("Neuron ID must be an integer >= 0")
        else:
            self.id = id

        if type not in (Neuron.I, Neuron.E):
             raise NeuronException("Neuron type must be -1 or 1 (inhibitory, excitatory")
        else:
            self.type = type

        self.threshold = max(Neuron.MIN_THRESHOLD, threshold)

        if not weight_delta > 0:
            raise NeuronException("Neuron weight_delta must be > 0")
        else:
            self.weight_delta = weight_delta

        if not decay_rate >= 0:
            raise NeuronException("Neuron decay_rate must be >= 0")
        else:
            self.decay_rate = decay_rate

        self.value = 0
        self.fired = False
        self.delta = 0
        self.upstream = {}
        self.downstream = []
        self.actors = []


    def __repr__(self):
        return "Neuron {id}, {type}: network: {n}, threshold: {t}, value: {v}".format(id=self.id, type={self.type}, n=self.network_name, t=self.threshold, v=self.value)

    def __eq__(self, other):
        return isinstance(other, Neuron) and other.network_name == self.network_name and other.id == self.id

    def add_upstream(self, id, weight):
        self.upstream[id] = weight

    def add_downstream(self, neuron):
        self.downstream.append(neuron)

    def excite(self, id):
        weight = self.upstream[id]
        self.delta = self.delta + weight
        self.actors.append(id)

    def inhibit(self, id):
        weight = self.upstream[id]
        self.delta = self.delta - weight
        self.actors.append(id)

    def integrate(self):
        self.value = self.value + self.delta

    def fire(self):
        if self.value > self.threshold:
            self.fired = True
            for n in self.downstream:
                if self.type == Neuron.E:
                    n.excite(self.id)
                else:
                    n.inhibit(self.id)
            self.value = 0
        else:
            self.fired = False
        self.value = max(0, self.value - self.decay_rate)
        self.update_weights()

        # reset neuron
        self.delta = 0
        self.actors = []

    def update_weights(self):
        # spike time dependent plasticity
        for id in self.upstream:
            weight = self.upstream[id]
            if id in self.actors:
                weight = min(1.0, weight + self.weight_delta)
            else:
                weight = max(1e-6, weight - self.weight_delta)
            self.upstream[id] = weight

    def set_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class NeuronException(Exception):
    pass


