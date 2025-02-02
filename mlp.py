import random
import math
from engine import Value


class Neuron:
    def __init__(self, ninputs):
        self.ninputs = ninputs
        self.w = [Value(random.uniform(-1, 1)) for _ in range(self.ninputs)]
        self.b = Value(random.uniform(-1, 1))
        self.o = Value(None)
        self.LR = 0.01

    def __call__(self, x) -> Value:
        act = sum((input * weight for input, weight in zip(x, self.w)), self.b)
        self.o = act.relu()
        return self.o

    def update(self):
        for w in self.w:
            w.value -= self.LR * w.grad

        self.b.value -= self.LR * self.b.grad

    def zero_grad(self):
        for w in self.w:
            w.grad = 0.0
        self.b.grad = 0.0


class Layer:
    def __init__(self, ninputs, noutputs):
        self.neurons = [Neuron(ninputs) for _ in range(noutputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def update(self):
        for n in self.neurons:
            n.update()

    def zero_grad(self):
        for n in self.neurons:
            n.zero_grad()


class MLP:
    def __init__(self, num_n_per_layer):
        self.layers = [
            Layer(num_n_per_layer[i], num_n_per_layer[i + 1])
            for i in range(len(num_n_per_layer) - 1)
        ]

    def __call__(self, x):
        curr_x = x
        for layer in self.layers:
            o = layer(curr_x)
            curr_x = o

        return o

    def update(self):
        for layer in self.layers:
            layer.update()

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()


# Make sure to reset the gradient to zero after each iteration
# It is ok to have a batch size, to expand parallel graphs and then accumulate
# the gradients at the end
# Make sure to use the gradients of the parameter to update and not anything else.
