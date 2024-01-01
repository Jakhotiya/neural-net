import math
class Value:

    def __init__(self, data, _children=(), _op='', _label=''):
        self.data = data
        self.prev = set(_children)
        self._label = _label
        self._grad = 0.0
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(label={self._label} data={self.data}, grad={self._grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self._grad += out._grad
            other._grad += out._grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        out._label = f"{self._label} * {other._label}"

        def _backward():
            self._grad += other.data * out._grad
            other._grad += self.data * out._grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __truediv__(self, other):  # self/other
        return self * (other ** -1)

    def __neg__(self):
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # self - other
        return self + (-other)

    def __pow__(self, power, modulo=None):
        assert isinstance(power, (int, float))
        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            self._grad += power * (self.data ** (power - 1)) * out._grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self._grad += math.exp(x) * out._grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self._grad += (1 - t ** 2) * out._grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self._grad += (out.data > 0) * out._grad

        out._backward = _backward

        return out

    def backward(self):
        """ This is topological sort of the graph."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self)
        self._grad = 1.0
        for node in reversed(topo):
            node._backward()


