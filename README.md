# Backpropagation Implementation

A from-scratch implementation of neural networks with backpropagation algorithm in Python. This project demonstrates how neural networks learn through gradient descent by implementing the core mathematical concepts without using external deep learning frameworks.

## Overview

This repository contains a complete implementation of:
- **Automatic Differentiation**: A `Value` class that tracks computational graphs and computes gradients using the chain rule
- **Neural Network Components**: Neurons, layers, and multi-layer perceptrons (MLP)
- **Training Loop**: Forward pass, loss computation, backpropagation, and gradient descent optimization

## Key Components

### Value Class
The `Value` class is the foundation of this implementation. It wraps numerical data and tracks:
- The computational graph (which operations created it)
- Gradients for each variable
- Backward pass implementation using the chain rule

Supported operations include:
- Addition and multiplication
- Exponentiation and division
- Hyperbolic tangent (tanh) activation function
- Custom gradient computation for each operation

### Neural Network Architecture

**Neuron**: A single neuron that computes `w * x + b` and applies tanh activation

**Layer**: A collection of neurons that transforms input through multiple parallel computations

**MLP (Multi-Layer Perceptron)**: A feedforward network composed of multiple layers

```python
# Create a network with 3 inputs, two hidden layers of 4 neurons each, and 1 output
n = MLP(3, [4, 4, 1])
```

## Training Process

The training loop consists of four main steps:

1. **Forward Pass**: Compute predictions by passing inputs through the network
2. **Loss Computation**: Calculate MSE (Mean Squared Error) loss
3. **Backward Pass**: Compute gradients using backpropagation
4. **Gradient Descent**: Update parameters in the direction opposite to gradients

```python
for epoch in range(10):
    # Forward pass
    predictions = [n(x) for x in inputs]
    loss = sum((pred - target)**2 for pred, target in zip(predictions, targets))
    
    # Backward pass
    for param in n.parameters():
        param.grad = 0.0
    loss.backward()
    
    # Update parameters
    for param in n.parameters():
        param.data += -0.05 * param.grad  # 0.05 is the learning rate
```

## How It Works

### Automatic Differentiation
Each operation (addition, multiplication, etc.) defines how gradients flow backward through it. For example:
- Addition: gradients flow equally to both inputs
- Multiplication: gradient to each input is scaled by the other input's value
- Tanh: gradient is scaled by `(1 - tanh(x)²)`

### Topological Ordering
The backward pass uses topological sorting to ensure gradients are computed in the correct order, respecting the computational dependencies.

### Gradient Descent
Parameters are updated by moving in the direction opposite to the gradient, scaled by a learning rate. This gradually reduces the loss, improving the network's predictions.

## Files

- `backpropagation_implementation.ipynb`: Complete Jupyter notebook with implementation and examples

## Requirements

- Python 3.x
- Standard library only (math, random)

## Usage

Open the notebook in Jupyter and run the cells sequentially. The notebook includes:
1. Core `Value` class implementation
2. `Neuron`, `Layer`, and `MLP` class definitions
3. Example training on sample data

## Learning Outcomes

This project illustrates:
- How neural networks compute gradients using the chain rule
- The importance of automatic differentiation in deep learning
- How gradient descent optimizes network parameters
- The mathematical foundation behind popular frameworks like PyTorch and TensorFlow

## References

This implementation is inspired by neural network fundamentals and demonstrates concepts from:
- Calculus (chain rule, partial derivatives)
- Linear algebra (vectors, matrices)
- Optimization (gradient descent)

