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

        self.population = 50 * (np.random.rand(self.population_size, self.dim) - 0.5)

        self.iter = 0  # current number of iteration
        self.mutation_success_counter = 0

        self.evaluator = evaluator
        self.population_analyzer = population_analyzer
        

    def Solve(self):
        for self.iter in range(self.iterations):
            # Analyze clusters in Population     
            if not self.iter % self.iterations_per_analysis:    
                self.population_analyzer.AnalyzePopulationAndSaveToLog(self.population, self.iter)

            # The population is evaluated (selection). The best adapted individuals take part in the reproduction process
            self.Reproduction()

            self.Crossover()
            self.Mutate()

            self.Succession()


    def Reproduction(self):
        fitness = np.apply_along_axis(self.evaluator.evaluate, 1, self.population).flatten()
        tmpPopulation = np.zeros(shape = self.population.shape)

        for i in range(self.population_size):
            tournamentIndices = np.random.randint(self.population_size, size = self.tournament_size)
            tmpfitness = np.zeros(shape = fitness.shape)
            tmpfitness[tournamentIndices] = fitness[tournamentIndices]
            
            tmpPopulation[i] = self.population[np.argmax(tmpfitness)]

        self.population = tmpPopulation.copy()


    def Mutate(self):
        mean_value_before = np.mean(np.apply_along_axis(self.evaluator.evaluate, 1, self.population))
        self.population = self.population + self.mutation_sigma * np.random.normal(size = self.population.shape)
        mean_value_after = np.mean(np.apply_along_axis(self.evaluator.evaluate, 1, self.population))

        if mean_value_after > mean_value_before:
            self.mutation_success_counter += 1

        if self.mutation_success_counter > 0.2 * self.iter:
            self.mutation_sigma *= 1.3
        else:    
            self.mutation_sigma /= 1.3

    def Crossover(self):
        number_of_crossovers = int(self.crossing_probability * self.population_size)
            
        # In each crossover, we select two parents
        # parents are deleted from population, their child is added to population
        # this phase SHRINKS population, population will regain its original size on succession

        rows_to_delete = []
        children = np.zeros(shape = (number_of_crossovers, self.population.shape[1]))

        for i in range(number_of_crossovers):
            first_parent_ind = np.random.randint(self.population_size)
            second_parent_ind = np.random.randint(self.population_size)

            rows_to_delete.append(first_parent_ind)
            rows_to_delete.append(second_parent_ind)

            child = (self.population[first_parent_ind] + self.population[second_parent_ind]) / 2
            children[i] = child


        self.population = np.delete(self.population, rows_to_delete, axis = 0)
        self.population = np.append(self.population, children, axis = 0)


    def Succession(self):
        # generative succession - we are taking elements after mutation
        # we keep population size constant, that's why we are duplicatind some elements sometimes
        add_individuals = np.random.choice(self.population.shape[0], self.population_size - self.population.shape[0])
        self.population = np.append(self.population, self.population[add_individuals], axis = 0)
