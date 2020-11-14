import argparse
import os
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from Evaluator import Evaluator
from Analyzer import Analyzer

parser = argparse.ArgumentParser(description='Evolutionary algorithm')
parser.add_argument('--goal_function', type=str, choices=['constant', 'gaussian', 'two_gaussians', 'rastrigin'], default='constant')

parser.add_argument('--A', type=float, default=10.0)
parser.add_argument('--mu1', type=float, default=0.0)
parser.add_argument('--sigma1', type=float, default=0.8)
parser.add_argument('--mu2', type=float, default=4.0)
parser.add_argument('--sigma2', type=float, default=1.0)

parser.add_argument('--population_size', type=int, default=200)
parser.add_argument('--dimension', type=int, default=1)
parser.add_argument('--mutation_sigma', type=float, default=1.0)
parser.add_argument('--tournament_size', type=int, default=4)
parser.add_argument('--crossing_probability', type=float, default=0.7)
parser.add_argument('--iterations', type=int, default=100)

parser.add_argument('--iterations_per_analysis', type=int, default=10)

parser.add_argument('--use_one_fifth_success_rule', type=eval, choices=[True, False],  default='False')

parser.add_argument('--log_dir', type=str, default="log")
parser.add_argument('--rep', type=int, default=0)

args = parser.parse_args()

print("use_one_fifth_success_rule", args.use_one_fifth_success_rule)


if (args.log_dir == "log"):
    if (args.goal_function == "gaussian"):
        arg_str = f"mu_{args.mu1}_sigma_{args.sigma1}"
    elif (args.goal_function == "two_gaussians"):
        arg_str = f"mu1_{args.mu1}_sigma1 {args.sigma1}_mu2_{args.mu2}_sigma2_{args.sigma2}"
    elif (args.goal_function == "rastrigin"):
        arg_str = f"A_{args.A}"
    else:
        arg_str = ""

    if (arg_str == ""):
        log_dir = os.path.join("results", f"{args.goal_function}")
    else:
        log_dir = os.path.join("results", f"{args.goal_function}_{arg_str}")

    log_dir = f"{log_dir}_dim_{args.dimension}_tournament_{args.tournament_size}"

# algorithm repetitions
for i in range(1, 4):
    args.log_dir = f"{log_dir}_rep_{i}"
    evaluator = Evaluator(args)

    analyzer = Analyzer(evaluator, args.log_dir, 5, should_plot=(i==1))

    algorithm = EvolutionaryAlgorithm(args, evaluator, analyzer)
    algorithm.Solve()