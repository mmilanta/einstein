from src.tensor import Tensor
from src.gradient import GTensor, ReLU
import random

class MLP():
    def __init__(self, hidden_sizes: list[int], start_size: int, end_size: int):
        self.weights: list[GTensor] = []
        start_sizes = [start_size] + hidden_sizes
        end_sizes = hidden_sizes + [end_size]
        for s, e in zip(start_sizes, end_sizes):
            self.weights.append(GTensor(Tensor(data=[random.random() for _ in range(s * e)], dims=(s, e))))
    def forward(self, x: GTensor) -> GTensor:
        activations: list[GTensor] = [x]
        for i, hidden_layer in enumerate(self.weights):
            new_activations = GTensor.einsum("ij,jk->ik", activations[-1], hidden_layer)
            activations.append(new_activations)
            if i < len(self.weights) - 1: # skip last step
                activations.append(ReLU(activations[-1]))
        return activations[-1]

    def reset(self):
        for w in self.weights:
            w.reset()

    def apply_gradient(self, factor: float):
        for w in self.weights:
            w._data = Tensor.add(w._data, Tensor.time_constant(w._grad, factor))
