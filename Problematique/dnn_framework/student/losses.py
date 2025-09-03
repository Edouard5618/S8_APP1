import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        N = target.shape[0]
        softmaxed_input = softmax(x)
        target_onehot = np.eye(x.shape[1])[target]

        # Compute cross-entropy loss
        L = -np.sum(target_onehot * np.log(softmaxed_input), axis=1)  # sum over classes
        loss = np.mean(L)  # average over batch

        # Gradient
        input_grad = (softmaxed_input - target_onehot) / N

        return loss, input_grad


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return y



class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        N = x.size
        L = np.sum((x-target)**2)/N
        dL_dx = 2/N*(x-target)
        return L, dL_dx
