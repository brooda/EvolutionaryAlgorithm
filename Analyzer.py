import time
import hashlib
import scipy
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import os
import pickle
import sys
from Evaluator import Evaluator

# this can be used for counting number of clusters
def gap_scores(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        ### https://stackoverflow.com/a/50804098/4793865
        cluster_to_representants = {i: np.where(km.labels_ == i)[0] for i in range(km.n_clusters)}

        cardinality_of_clusters = {}
        std_in_clusters = {}

        sum_cardinality = 0
        sum_std = 0
        
        for key, value in cluster_to_representants.items():
            cardinality = value.size
            std = np.std(data[value], axis = 0)
            
            sum_std += np.sum(std) * cardinality
            sum_cardinality += cardinality


        std_normalized = sum_std / sum_cardinality


        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        if k < 4:
            gap = -10000


        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

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

                # bad code, change that
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
        