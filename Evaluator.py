import numpy as np

# Gives 
class Evaluator:
    def __init__(self, args):
        self.goal_function = args.goal_function
        self.A = args.A
        self.mu1 = args.mu1
        self.sigma1 = args.sigma1
        self.mu2 = args.mu2
        self.sigma2 = args.sigma2

    def constant(self, x):
        return 1

    def gaussian(self, x):
        return np.sum(np.exp(-np.power(x - self.mu1, 2.) / (2 * np.power(self.sigma1, 2.))))

    def two_gaussians(self, x):
        return np.sum(np.exp(-np.power(x - self.mu1, 2.) / (2 * np.power(self.sigma1, 2.)))) + \
            np.sum(np.exp(-np.power(x - self.mu2, 2.) / (2 * np.power(self.sigma2, 2.))))

    def rastrigin(self, x):
        n = x.shape[0]
        return self.A*n + np.sum(x - self.A * np.cos(2 * np.pi * x))

    def evaluate(self, x):
        if self.goal_function == 'constant':
            return self.constant(x)
        elif self.goal_function == 'gaussian':
            return self.gaussian(x)
        elif self.goal_function == 'two_gaussians':
            return self.two_gaussians(x)
        elif self.goal_function == 'rastrigin':
            return self.rastrigin(x)
        else:
            raise ValueError("incorrect goal function")
