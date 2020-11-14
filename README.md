# Implementation of classical genetic algoritm with interface for inspecting populations dynamically

In `doc` folder there is a report about this project. The following description is about content of `src` directory.

## Files:
* `EvolutionaryAlgorithm.py` - implementation of GA
* `Evaluator.py` - goal functions and plotting
* `Analyzer.py` - class for statistical analysis of clusters in subsequent iterations. Analysis is dumped to pickle binary files


## Arguments:
* `--goal_function, type=str, choices=['constant', 'gaussian', 'two_gaussians', 'rastrigin'], default='constant')`
* `--A', type=float, default=10.0`
* `--mu1', type=float, default=0.0`
* `--sigma1', type=float, default=1.0`
* `--mu2', type=float, default=4.0`
* `--sigma2', type=float, default=1.0`
* `--population_size', type=int, default=1000`
* `--dimension', type=int, default=1`
* `--mutation_sigma', type=float, default=1.0`
* `--tournament_size', type=int, default=4`
* `--crossing_probability', type=float, default=0.5`
* `--iterations', type=int, default=200`
* `--iterations_per_analysis', type=int, default=10`
* `--log_file', type=str, default="log.txt"`

Examples of how to run the program are in bash files (test[1, 2, 3, 4, 5].sh)


## Organization of expetiment pipeline
Each execution of program mean 3 independent runs of algorithm. In `results` folder, subfolders for each of runs are created.

Then, in each subfolder, next subfolders are created:
* `clusters=1`
* `clusters=2`
* `clusters=3`
* `clusters=4`
* `clusters=5`

Each of those subfolders corresponds to output of clustering for different numbers of clusters.

Then, in those folder, two folders are created:
* `images`: for first run of algorithm and for case when domain of function is of dimension 1 or 2, for each iteration function with population is plotted. It is worth noting, that plotting images is the most time consuming element of all pipeline
* `statistics`: for each iterations statistics about clusters are dumped to those statistics are: cardinality of clusters, means of clusters, standard deviations in clusters

## Analysis
In `results/analysis` folder, there is notebook for plotting further needed graphs. More information about it can be achieved upon request.