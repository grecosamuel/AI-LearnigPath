# Adaline GD Model

### Introduction and general logic

Adaline is a learning algorithm published by Bernard Widrow and Tedd Hoff, the name describe the algorithm as **Adaptive Linear Neuron**. There is a main difference between the previous algorithm (Perceptron) because Adaline trying to define and minimize the cost functions, as a plus instead of update weights with unit step function use a linear activation function that is network input's identity.

To minimize the cost functions Adaline use the gradient descent logic.

### Logical steps

It is possible to implement the Adaptive Linear Neuron Algorithm with the next steps:

1. Initialize vector of weigths to 0
2. Initialize empty list for save the costs
3. For each epoch defined:
   1. Get the output with scalar product between matrix of data and vector of weights without bias
   2. Check the error using the difference between the correct value and the output obtained previously
   3. Update the weights doing the multiplication of the learning rate and the transposed data matrix with the scalar product of the errors
   4. Update the bias with multiplication of learning rate and SSE (Sum of Squared Errors)

---

Thanks to [Sebastian Raschka](https://github.com/rasbthttps:/) and his book **Machine Learning with Python**
