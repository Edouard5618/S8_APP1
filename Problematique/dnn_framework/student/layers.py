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

        input_grad = (output_grad@self.weights)
        w_grad = (x.T@output_grad).T
        b_grad = np.sum(output_grad, axis=0)

        param_grad = {'w':w_grad,'b':b_grad}
        return input_grad, param_grad


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        self.gamma = np.random.rand(input_count)
        self.beta = np.random.rand(input_count)

    def get_parameters(self):
        params = {"gamma":self.gamma,"beta":self.beta}
        return params

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        epsilon = 1e-7 # Avoid division by zero
        M = x.shape[0] # Batch size

        u = (1/M)*np.sum(x,axis=0)
        sigma_2 = (1/M)*np.sum((x-u)**2, axis=0)
        x_ = (x-u)/np.sqrt(sigma_2+epsilon)
        y = x_ * self.gamma + self.beta

        cache = {"x":x,"gamma":self.gamma}
        return y, cache

    def _forward_training(self, x):
        raise NotImplementedError()

    def _forward_evaluation(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()
