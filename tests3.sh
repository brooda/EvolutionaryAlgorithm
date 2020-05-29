for tournament in 4
do
    python main.py --goal_function constant --iterations_per_analysis 1 --tournament_size $tournament --rep 0 --dimension 1 &   
    python main.py --goal_function gaussian --iterations_per_analysis 1 --tournament_size $tournament --rep 0 --dimension 1 & 
    python main.py --goal_function two_gaussians --iterations_per_analysis 1 --tournament_size $tournament --rep 0 --dimension 1 & 
    python main.py --goal_function rastrigin --iterations_per_analysis 1 --tournament_size $tournament --rep 0 --dimension 1 & 
    python main.py --goal_function constant --iterations_per_analysis 1 --tournament_size $tournament --rep 0 --dimension 2 & 
    python main.py --goal_function gaussian --iterations_per_analysis 1 --tournament_size $tournament --rep 0 --dimension 2 & 
    python main.py --goal_function two_gaussians --iterations_per_analysis 1 --tournament_size $tournament --rep 0 --dimension 2 & 
    python main.py --goal_function rastrigin --iterations_per_analysis 1 --tournament_size $tournament --rep 0 --dimension 2
done