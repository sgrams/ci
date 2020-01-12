#!/bin/env python
# 2020/01/12
# Forward pass - neural network: activation function
# Stanislaw Grams <sjg@fmdx.pl>
# 07-neural_networks/01-forward_pass.py
import numpy as np

def nonlinear (x,deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp (-x))

def forward_pass (age, weight, height):
    hidden1 = nonlinear (0.80109 + (-0.46122 * age) + (0.97314 * weight) + (-0.39203 * height))
    hidden2 = nonlinear (0.43529 + (2.10584 * weight) + (-0.57847 * height) + (0.78548 * age))

    return (-0.81546 * hidden1) +  (1.03775 * hidden2) - 0.2368

test_val = forward_pass (23, 75, 176)
print (test_val)
