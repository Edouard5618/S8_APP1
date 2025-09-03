from random import betavariate

import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        self.weights = np.random.rand(output_count,input_count)
        self.bias = np.random.rand(output_count)

    def get_parameters(self):

        parameters = {"w": self.weights, "b":self.bias}
        return parameters

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        y = x@self.weights.T+self.bias
        cache = {"x":x,"w":self.weights}
        return y,cache

    def backward(self, output_grad, cache):
        x = cache['x'][:]

        input_grad = output_grad @ self.weights
        w_grad = output_grad.T @ x
        b_grad = np.sum(output_grad, axis=0)

        param_grad = {'w':w_grad,'b':b_grad}
        return input_grad, param_grad


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """


    def __init__(self, input_count, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = np.random.rand(input_count)
        self.beta = np.random.rand(input_count)
        self.global_mean = np.zeros(input_count)
        self.global_variance = np.zeros(input_count)
        self.epsilon = 1e-7

    def get_parameters(self):
        params = {"gamma":self.gamma,"beta":self.beta}
        return params

    def get_buffers(self):
        buffers = {"global_mean":self.global_mean, "global_variance": self.global_variance}
        return buffers

    def forward(self, x):
        if self.is_training():
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)


    def _forward_training(self, x):
        M = x.shape[0]  # Batch size

        u = (1 / M) * np.sum(x, axis=0)
        sigma_2 = (1 / M) * np.sum((x - u) ** 2, axis=0)
        x_ = (x - u) / np.sqrt(sigma_2 + self.epsilon)
        y = x_ * self.gamma + self.beta

        self.global_mean = (1 - self.alpha) * self.global_mean + self.alpha * u
        self.global_variance = (1 - self.alpha) * self.global_variance + self.alpha * sigma_2

        cache = {"x": x, "x_":x_, "gamma": self.gamma, "beta": self.beta, "u": u, "sigma_2": sigma_2, "M": M}
        return y, cache

    def _forward_evaluation(self, x):
        x_ = (x - self.global_mean) / np.sqrt(self.global_variance)
        y = x_ * self.gamma + self.beta
        cache = {"x":x,"gamma":self.gamma}
        return y, cache

    def backward(self, output_grad, cache):
        x = cache['x'][:]
        x_ = cache['x_'][:]
        gamma = cache['gamma'][:]
        beta = cache['beta'][:]
        u = cache['u'][:]
        sigma_2 = cache['sigma_2'][:]
        M = cache['M']


        x__grad = output_grad * gamma
        sigma_2_grad = np.sum(x__grad*(x-u)*(-0.5*(sigma_2+self.epsilon)**(-3/2)) ,axis=0)
        u_grad = -np.sum(x__grad/np.sqrt(sigma_2+self.epsilon), axis=0)
        x_grad = x__grad/np.sqrt(sigma_2+self.epsilon)+(2/M)*sigma_2_grad*(x-u)+(1/M)*u_grad
        gamma_grad = np.sum(output_grad*x_, axis=0)
        beta_grad = np.sum(output_grad, axis=0)

        input_grad = x_grad
        param_grad = {"gamma":gamma_grad,"beta":beta_grad}

        return input_grad, param_grad


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        y = 1/(1+np.exp(-x))
        cache = {"x": x}
        return y, cache

    def backward(self, output_grad, cache):
        x = cache["x"][:]
        y = 1/(1+np.exp(-x))
        x_grad = output_grad*(1-y)*y
        return x_grad, x


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        y = np.maximum(x,0)
        cache = {"x":x}
        return y, cache

    def backward(self, output_grad, cache):
        x = cache["x"][:]
        x = (x>0).astype(float)
        input_grad = x*output_grad
        return input_grad, cache
