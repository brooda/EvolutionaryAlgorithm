import numpy as np
from Evaluator import Evaluator

class EvolutionaryAlgorithm:
    def __init__(self, args, evaluator, population_analyzer):
        self.population_size = args.population_size
        self.dim = args.dimension
        self.mutation_sigma =args.mutation_sigma
        self.tournament_size = args.tournament_size
        self.crossing_probability = args.crossing_probability
        self.iterations = args.iterations

        self.iterations_per_analysis = args.iterations_per_analysis

        if args.goal_function == "rastrigin":
            population_multiplier = 20
        else:
            population_multiplier = 20

        self.population = population_multiplier * (np.random.rand(self.population_size, self.dim) - 0.5)

        self.iter = 0  # current number of iteration
        self.mutation_success_counter = 0

        self.evaluator = evaluator
        self.population_analyzer = population_analyzer
        
        self.use_one_fifth_success_rule = args.use_one_fifth_success_rule


    def Solve(self):
        for self.iter in range(self.iterations):
            # Analyze clusters in Population     
            if not self.iter % self.iterations_per_analysis:    
                self.population_analyzer.AnalyzePopulationAndSaveToLogAndDrawPlotOfPopulation(self.population, self.iter)

            number_of_crossovers = self.Reproduction()
            self.Crossover(number_of_crossovers)
            self.Mutate(number_of_crossovers)
            self.Succession()


    def Reproduction(self):
        fitness = np.apply_along_axis(self.evaluator.evaluate, 1, self.population).flatten()
        
        # Number of crossings for next phase is calculated here, because at each crossing child element is created from two parents. 
        # Thats why output population is bigger than input population
        # Elements that will be used in crossing are kept at the end of list 
        # Of course, elements for crossing are selected randomly, but that random selection is implemented in reproduction phase
        # The fact, that elements for crossing are in the end of list are just for convenient program organization.     
        
        number_of_crossings = np.sum(np.random.uniform(size=self.population_size) < self.crossing_probability)
        tmpPopulation = np.zeros(shape = (self.population_size + number_of_crossings, self.population.shape[1]))

        for i in range(tmpPopulation.shape[0]):
            tournamentIndices = np.random.randint(self.population_size, size = self.tournament_size)
            tmpfitness = np.zeros(shape = fitness.shape)
            tmpfitness[tournamentIndices] = fitness[tournamentIndices]

            tmpPopulation[i] = self.population[np.argmax(tmpfitness)]

        self.population = tmpPopulation.copy()
        return number_of_crossings


    def Crossover(self, number_of_crossovers):
        for i in range(number_of_crossovers):
            parent_1 = self.population[self.population_size - number_of_crossovers + 2*i]
            parent_2 = self.population[self.population_size - number_of_crossovers + 2*i + 1]
            child = (parent_1 + parent_2) / 2
            self.population[self.population_size - number_of_crossovers + i] = child

        self.population = self.population[:self.population_size, ]


    def Mutate(self, number_of_crossovers):
        mean_value_before = np.mean(np.apply_along_axis(self.evaluator.evaluate, 1, self.population))

        # mutation is applied only to element that come from crossover  
        self.population[-number_of_crossovers:] = self.population[-number_of_crossovers:] + \
                                                  self.mutation_sigma * np.random.normal(size = (number_of_crossovers, self.population.shape[1]))
        
        mean_value_after = np.mean(np.apply_along_axis(self.evaluator.evaluate, 1, self.population))

        if self.use_one_fifth_success_rule:
            if mean_value_after > mean_value_before:
                self.mutation_success_counter += 1

            if self.mutation_success_counter > 0.2 * self.iter:
                self.mutation_sigma *= 1.3
            else:    
                self.mutation_sigma /= 1.3

            
    def Succession(self):
        # generative succession - we do nothing
        pass