import time
import hashlib
import scipy
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
import pickle
import sys
from Evaluator import Evaluator

class Analyzer:
    def __init__(self, evaluator, log_dir, max_clusters, should_plot):
        self.log_dir = log_dir
        self.max_clusters = max_clusters

        self.statistics_dir = os.path.join(self.log_dir, "statistics")
        self.images_dir = os.path.join(self.log_dir, "images")

        self.evaluator = evaluator

        self.should_plot = should_plot

        try:
            os.mkdir(self.log_dir)
            os.mkdir(self.statistics_dir)
            os.mkdir(self.images_dir)

            for i in range(1, self.max_clusters + 1):
                os.mkdir(os.path.join(self.statistics_dir, f"clusters={i}"))
                os.mkdir(os.path.join(self.images_dir, f"clusters={i}"))

        except OSError:
            print ("Creation of the directory %s failed" % self.log_dir)
        else:
            print ("Successfully created the directory %s " % self.log_dir)
            

    def AnalyzePopulationAndSaveToLogAndDrawPlotOfPopulation(self, population, iteration):
        for number_of_clusters in range(1, self.max_clusters + 1):
            statistics, population_dict = self.AnalyzePopulation(population, iteration, number_of_clusters)
            
            if self.should_plot:
                self.evaluator.plot(os.path.join(self.images_dir, f"clusters={number_of_clusters}"), population_dict, iteration)

                # on 0th iteration also plot without population is plotted
                if (iteration == 0):
                    self.evaluator.plot(os.path.join(self.images_dir, f"clusters={number_of_clusters}"))

            with open(os.path.join(self.statistics_dir, f"clusters={number_of_clusters}", f"{iteration}"), 'wb') as outfile:
                pickle.dump(statistics, outfile)


    def AnalyzePopulation(self, population, iteration, k):
        statistics = {}
        statistics["iteration"] = iteration
        statistics["mean"] = np.mean(population, axis = 0)
        statistics["std"] = np.std(population, axis = 0)

        statistics["optimal_k"] = k

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(population)
        
        statistics["cluster_centers"] = kmeans.cluster_centers_

        ### https://stackoverflow.com/a/50804098/4793865
        cluster_to_representants = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

        cardinality_of_clusters = {}
        std_in_clusters = {}

        for k, v in cluster_to_representants.items():
            cardinality_of_clusters[k] = v.size
            std_in_clusters[k] = np.std(population[v], axis = 0)

        ks = np.array(list(cardinality_of_clusters.keys()))
        vs = np.array(list(cardinality_of_clusters.values()))
               
        sorted_inds = ks[np.flip(np.argsort(vs))]

        population_pieces_descending_cardinality = []
        for ind in sorted_inds:
            population_pieces_descending_cardinality.append(population[cluster_to_representants[ind]])

        statistics["cardinality_of_clusters"] = cardinality_of_clusters
        statistics["std_in_clusters"] = std_in_clusters

        return statistics, population_pieces_descending_cardinality
        