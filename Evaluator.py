import numpy as np
import matplotlib.pyplot as plt
import os

# Gives 
class Evaluator:
    def __init__(self, args):
        self.goal_function = args.goal_function
        self.A = args.A
        self.mu1 = args.mu1
        self.sigma1 = args.sigma1
        self.mu2 = args.mu2
        self.sigma2 = args.sigma2

        self.log_dir = args.log_dir
        self.dim = args.dimension


    def constant(self, x):
        return 1


    def gaussian(self, x):
        return np.sum( (1.0/self.sigma1) * np.exp(-np.power(x - self.mu1, 2.) / (2 * np.power(self.sigma1, 2.))))


    def two_gaussians(self, x):
        return np.sum( (1.0/self.sigma1) * np.exp(-np.power(x - self.mu1, 2.) / (2 * np.power(self.sigma1, 2.)))) + \
            np.sum( (1.0/self.sigma2) * np.exp(-np.power(x - self.mu2, 2.) / (2 * np.power(self.sigma2, 2.))))


    def rastrigin(self, x):
        n = x.shape[0]
        return - (self.A*n + np.sum(np.square(x) - self.A * np.cos(2 * np.pi * x)))


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


    def plot(self, population = None, iteration = None):
        if self.dim > 2:
            raise Exception("dimension of problem too big")


        if self.dim == 1:
            evaluate_vect = np.vectorize(self.evaluate)
            x = np.linspace(-10, 10, 500)
            y = evaluate_vect(x)

            plt.clf()
            plt.plot(x, y, color='b')

            if population is None:
                plt.savefig(os.path.join(self.log_dir, "images", "function.png"))
            else:
                plt.scatter(population, evaluate_vect(population), color='r', s=5)
                plt.savefig(os.path.join(self.log_dir, "images", f"{iteration}.png")) 

        elif self.dim == 2:            
            evaluate_vect = np.vectorize(self.evaluate, signature='(n)->()')
            
            x = np.linspace(-10, 10, 500)
            y = np.linspace(-10, 10, 500)

            grid = np.array(np.meshgrid(x, y))
            grid = np.moveaxis(grid, 0, -1)
            grid = np.moveaxis(grid, 0, 1)

            z = evaluate_vect(grid)

            plt.clf()
            cp = plt.contourf(x, y, z)
            plt.colorbar(cp)
            
            if population is None:
                plt.savefig(os.path.join(self.log_dir, "images", "function.png"))
            else:
                plt.scatter(population[:, 0], population[:, 1], color='r', s=5)
                plt.savefig(os.path.join(self.log_dir, "images", f"{iteration}.png"))
