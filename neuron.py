import numpy as np

class Neuron:

    # Neuron Type
    I = -1.0 # Inhibitory
    E = 1.0 # Excitatory
    MIN_THRESHOLD = 1e-3
    MIN_DECAY_RATE = 1e-1
    MIN_WEIGHT_DELTA = 1e-3

    def __init__(self, type, threshold, position, weight_association_delta=(0.01, 0.01), weight_decay_delta=(0.04, 0.02), decay_rate=(0.5, 0.5)):

        self.type = type;
        self.threshold = max(Neuron.MIN_THRESHOLD, threshold)
        self.position = position # tuple (x, y, z)
        self.weight_association_delta = max(Neuron.MIN_WEIGHT_DELTA, weight_association_delta[1] * np.random.randn() + weight_association_delta[0])
        #self.weight_decay_delta = max(Neuron.MIN_WEIGHT_DELTA, weight_decay_delta[1] * np.random.randn() + weight_decay_delta[0])
        self.weight_decay_delta = self.weight_association_delta
        self.decay_rate = max(Neuron.MIN_DECAY_RATE, decay_rate[1] * np.random.randn() + decay_rate[0])
        self.value = 0
        self.fired = False
        self.delta = 0
        self.upstream = {}
        self.downstream = []
        self.actors = []

    def add_upstream(self, neuron, weight):
        self.upstream[neuron] = weight

    def add_downstream(self, neuron):
        self.downstream.append(neuron)

    def activate_directly(self, value):
        self.delta = self.delta + value

    def activate(self, actor, type):
        weight = self.upstream[actor]
        self.delta = self.delta + type * weight
        self.actors.append(actor)

    def integrate_and_fire(self):
        self.value = self.value + self.delta
        self.fired = False
        if self.value > self.threshold:
            for n in self.downstream:
                n.activate(self, self.type)
            self.value = 0
            self.fired = True
        self.value = max(0, self.value - self.decay_rate)
        self.delta = 0

    def update_weights(self):
        # spike time dependent plasticity
        for neuron in self.upstream:
            weight = self.upstream[neuron]
            if neuron in self.actors:
                weight = min(1.0, weight + self.weight_association_delta)
            else:
                weight = max(1e-6, weight - self.weight_decay_delta)
            self.upstream[neuron] = weight
        self.actors = []


