for dim in 1 3 5
do
    python main.py --goal_function constant --iterations_per_analysis 1 --tournament_size 2 --rep 0 --dimension $dim
    python main.py --goal_function gaussian --iterations_per_analysis 1 --tournament_size 2 --rep 0 --dimension $dim
    python main.py --goal_function two_gaussians --iterations_per_analysis 1 --tournament_size 2 --rep 0 --dimension $dim
    python main.py --goal_function rastrigin --iterations_per_analysis 1 --tournament_size 2 --rep 0 --dimension $dim
done