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
        if isinstance(x, (float)):
            n = 1
        else:
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

    # population is an array of elements in clusters sorted in decreasing order (cardinality)
    def plot(self, path_to_save = None, population = None, iteration = None):
        colors_of_clusters = ['r', 'g', 'y', 'm', 'k']
        # red green yellow magenta black

        if self.dim == 1:
            evaluate_vect = np.vectorize(self.evaluate)
            x = np.linspace(-10, 10, 500)
            y = evaluate_vect(x)

            plt.clf()
            plt.plot(x, y, color='b')

            if population is None:
                plt.savefig(os.path.join(path_to_save, "function.png"))
            else:
                for i, population_cluster in enumerate(population):
                    plt.scatter(population_cluster, evaluate_vect(population_cluster), color=colors_of_clusters[i], s=7)
                    plt.savefig(os.path.join(path_to_save, f"{iteration}.png")) 

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
                plt.savefig(os.path.join(path_to_save, "function.png"))
            else:
                for i, population_cluster in enumerate(population):
                    plt.scatter(population_cluster[:, 0], population_cluster[:, 1], color=colors_of_clusters[i], s=10)
                    plt.savefig(os.path.join(path_to_save, f"{iteration}.png"))
