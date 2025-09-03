from dnn_framework.optimizer import Optimizer


class SgdOptimizer(Optimizer):
    """
    This class implements a stochastic gradient descent optimizer.
    """

    def __init__(self, parameters, learning_rate=0.01):
        super().__init__(parameters)
        self.lr = learning_rate
        return

    def _step_parameter(self, parameter, parameter_grad, parameter_name):
        parameter -= self.lr*parameter_grad
        return parameter