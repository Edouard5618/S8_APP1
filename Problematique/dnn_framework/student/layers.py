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

        cache = {"x": x, "gamma": self.gamma}
        return y, cache

    def _forward_evaluation(self, x):
        x_ = (x - self.global_mean) / np.sqrt(self.global_variance)
        y = x_ * self.gamma + self.beta
        cache = {"x":x,"gamma":self.gamma}
        return y, cache

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
