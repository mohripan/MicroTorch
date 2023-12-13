import numpy as np

class MicroNode:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.grad = None
        self.requires_grad = requires_grad
        self.backward = lambda: None

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
        
    def _add(self, other):
        other = other if isinstance(other, MicroNode) else MicroNode(other)

        result = MicroNode(self.data + other.data)

        def _backward():
            if self.requires_grad:
                self.grad = np.ones_like(self.data)
            if other.requires_grad:
                other.grad = np.ones_like(other.data)

        result.backward = _backward

        return result

    def add(self, other):
        return self._add(other)
    
    def __add__(self, other):
        return self._add(other)

    def sub(self, other):
        other = other if isinstance(other, MicroNode) else MicroNode(other)
        return MicroNode(self.data - other.data)

    def mul(self, other):
        other = other if isinstance(other, MicroNode) else MicroNode(other)
        return MicroNode(self.data * other.data)

    def div(self, other):
        other = other if isinstance(other, MicroNode) else MicroNode(other)
        return MicroNode(self.data / other.data)