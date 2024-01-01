from random import random
from value import Value

class SingleInputNeuron():
    def __init__(self):
        self.weight = Value(5.0)
        self.bias = Value(0.0)

    def __call__(self, x):
        out =  self.weight * x + self.bias
        return out


"""
Via this Neuron we just try to fit our data to y = x
This is a good example to play around with activation function
loss function, learning rate and how they affect overall training. 
"""
n = SingleInputNeuron()

learning_rate = 0.05

for i in range(400):

    # prediction 1
    p1 = n(1)
    # expected 1
    e1 = 1

    # prediction 1
    p2 = n(2)
    # expected 1
    e2 = 2

    """ 
    Notice how the following loss function is of the form 
    y = a^2 + b^2 
    Graph of this function is a parabola.
    We choose parabolic function because it has a minima. We define our loss function
    in terms of parabola because then a minimum can be achived. Straight lines don't have a minimum. 
    Hence we don't choose linear function for our loss.
    """
    loss = ((p1-e1)**2 + (p2-e2)**2)

    n.weight._grad = 0
    n.bias._grad = 0
    loss.backward()

    """
    If you think about parabola, if you have negative slope, you want your x to increase
    and if you have positive slope you want your x to decrease. This becomes much clear
    if you the parabola and see what needs to be done in order to achieve local minima
    """
    n.weight.data += -learning_rate * n.weight._grad
    n.bias.data += -learning_rate * n.bias._grad


print("Result:",n(5))
print("Result:",n(11))
print("Result:",n(13))
print("Result:",n(80))
print("Result:",n(799))

