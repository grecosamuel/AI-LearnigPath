# Perceptron Model

### Introduction and general logic

This learning algorithm (ideated by Frank Rosenblatt) is based on MCP neuron by McCullock and Walter Pitts, where through a simple logi c gate it's possible to obtain a binary output.

Perceptron is considereted as a binary classification algorithm that use an activation function with linear combination of values in input and a vector of weights.

The logic behind saying that if the activation of a sample is higher than the threshold we can classify the sample with value 1, else we can classify with -1, this represent the general logic around binary classification.


### Logical steps

Approach used to ideate the use of MCP neuron and Perceptron Model consist to simulate the behavior of a single neuron of the brain.

It is possible to resume in a few steps:

1. Initialize vector of weigths to 0 or small random numbers.
2. For each training sample apply:
   1. Calculate output value
   2. Update weights


### Updating weights

About the vector of the weights, each weight will be updated with a new value obtained by the calculation of the learning rules of Perceptron.

A defined learning rate is multiplicated to the difference between the represented value of the right label and the represented value of the predicted label.

---

Thanks to [Sebastian Raschka](https://github.com/rasbthttps:/) and his book **Machine Learning with Python**
