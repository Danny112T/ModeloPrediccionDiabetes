import matplotlib.pyplot as plt
import numpy as np

def sigmod(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return (2/(1+np.exp(-2*x)))-1

""" x = np.linspace(-10,10,100)
y = sigmod(x)

plt.figure(figsize=(9,7))
plt.plot(x,y)
plt.title('Función Sigmoide')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True)
plt.show() """

def tanh(x):
    return (2/(1+np.exp(-2*x)))-1

""" x = np.linspace(-10,10,100)
y = tanh(x)

plt.figure(figsize=(8,6))
plt.plot(x,y)
plt.title('Función Tanh')
plt.xlabel('x')
plt.ylabel('Tanh(x)')
plt.grid(True)
plt.show() """

def ReLU(x):
    return np.maximum(0,x) 

""" x = np.linspace(-10,10,100)
y = ReLU(x)

plt.figure(figsize=(8,6))
plt.plot(x,y)
plt.title('Función ReLU')
plt.xlabel('x')
plt.ylabel('ReLu(x)')
plt.grid(True)
plt.show() """

def LeakyReLU(x,alpha):
    return np.maximum(alpha * x,x)

""" x = np.linspace(-10,10,100)
a = 0.01
y = LeakyReLU(x,a)

plt.figure(figsize=(8,6))
plt.plot(x,y)
plt.title('Función ReLU')
plt.xlabel('x')
plt.ylabel('ReLu(x)')
plt.grid(True)
plt.show() """

def elu(x, a):
    return np.where(x > 0, x, a * (np.exp(x) - 1))

""" x = np.linspace(-10,10,100)
a = 1
y = elu(x,a)

plt.figure(figsize=(8,6))
plt.plot(x,y)
plt.title('Función elu')
plt.xlabel('x')
plt.ylabel('Elu(x)')
plt.grid(True)
plt.show() """

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)

x = np.linspace(-10,10,100)
y = softmax(x)

plt.figure(figsize=(8,6))
plt.plot(x,y)
plt.title('Función softmax')
plt.xlabel('x')
plt.ylabel('softmax(x)')
plt.grid(True)
plt.show()
