from random import uniform

from value import Value

class Neuron():

    def __init__(self,nin):
        self.weights = [Value(uniform(-1,1)) for _ in range(0,nin)]
        self.bias = Value(uniform(-1,1))

    def __call__(self,x):
        activation =  sum((xi*wi for wi,xi in zip(self.weights,x)),self.bias)
        out = activation.tanh()
        return out

    def parameters(self):
        return self.weights + [self.bias]

"""
What happens when number of inputs to neuron
are higher than number of weights?
Are some inputs dropped
"""

class Layer():

    def __init__(self,nin,nout):
        """ nin in how many inputs does a single neuron take
            nout is how many neurons to create a layer
        """
        self.neurons = [Neuron(nin) for _ in range(0,nout)]

    def __call__(self,x):
        """ x is number of inputs the layer takes
            from last layer, or rather input data
        """

        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class Network():
    """
    This is also known as MLP multi-layer perceptron
    """
    def __init__(self,inputs_to_network,sizes_of_layers):

        # If number of initial inputs is k then dimention of single neuron in first layer
        # will be k
        # if we create first layer to be of m neurons then that m would be dimentions
        # of each neuron in next layer.
        # Here by dimention of neuron we mean axons/connections/weights on neuron
        sz = [inputs_to_network]+sizes_of_layers
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(sizes_of_layers))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]



