from neuron import *

class Sensor:

    MIN_THRESHOLD = 0.001

    def __init__(self, threshold, position, decay_rate=0.01):

        self.threshold = max(Neuron.MIN_THRESHOLD, threshold)
        self.position = position # tuple (x, y, z)
        self.decay_rate = decay_rate

        self.delta = 0
        self.value = 0
        self.fired = False
        self.downstream = []

    def add_downstream(self, neuron):
        self.downstream.append(neuron)

    def add_upstream(self, neuron, weight):
        pass

    def activate(self, value):
        self.delta = self.delta + value

    def integrate_and_fire(self):
        self.value = self.value + self.delta
        self.fired = False
        if self.value > self.threshold:
            for n in self.downstream:
                n.activate(self, Neuron.E)
            self.value = 0
            self.fired = True
        self.value = max(0, self.value - self.decay_rate)
        self.delta = 0



