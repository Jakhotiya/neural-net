# micrograd

This repository began with following Andrej Karpathy's lecture
on Youtube.

This is a precursor to building nano-gpt.

I learned how backpropogration is implemented
and how weights and biases are tweaked based on loss or error.

After coding along the lecture, I created single-neuron.py
file to better understand back-propogration and concepts
behind it.

I shall revisit my following questions later, after finishing up 
4 lectures by Andrej

1. In multi-layer perceptron how do you decide
how many layers do you need and what sizes?
2. How do you decide on input size and number of weights on single neuron?
3. How to choose which loss function to use?
4. How does training data available influence number of parameters? Does it affect number of parameters? 
5. What does an activation function really mean?

While playing around with the single neuron, I got some intuition
on how not to choose a loss function. You loss function must should have 
a minima. Preferable only one global minima.

Another thing is, since neurons are complex mathematical functions
you must always find a way to represent your data in terms of number. 
That matters a lot. 

I understood on high-level GPU's might be useful. I wrote a for loop
to train the network on data. For large data each iteration of the
loop will do computation. A lot of these computations can be run in parellel.

