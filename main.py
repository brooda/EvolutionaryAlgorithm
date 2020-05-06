import argparse
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from Evaluator import Evaluator
from Analyzer import Analyzer

parser = argparse.ArgumentParser(description='Evolutionary algorithm')
parser.add_argument('--goal_function', type=str, choices=['constant', 'gaussian', 'two_gaussians', 'rastrigin'], default='constant')

parser.add_argument('--A', type=float, default=10.0)
parser.add_argument('--mu1', type=float, default=0.0)
parser.add_argument('--sigma1', type=float, default=1.0)
parser.add_argument('--mu2', type=float, default=4.0)
parser.add_argument('--sigma2', type=float, default=1.0)

parser.add_argument('--population_size', type=int, default=1000)
parser.add_argument('--dimension', type=int, default=1)
parser.add_argument('--mutation_sigma', type=float, default=1.0)
parser.add_argument('--tournament_size', type=int, default=4)
parser.add_argument('--crossing_probability', type=float, default=0.5)
parser.add_argument('--iterations', type=int, default=200)

parser.add_argument('--iterations_per_analysis', type=int, default=10)

parser.add_argument('--log_file', type=str, default="log.txt")

args = parser.parse_args()

evaluator = Evaluator(args)
analyzer = Analyzer(args.log_file)

algorithm = EvolutionaryAlgorithm(args, evaluator, analyzer)
algorithm.Solve()