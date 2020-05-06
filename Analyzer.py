import time
import hashlib
import scipy
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

def optimalK(data, nrefs=3, maxClusters=15):
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
        
        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal





class Analyzer:
    def __init__(self, log_file):
        self.log_file = log_file


    def AnalyzePopulationAndSaveToLog(self, population, iteration):
        statistics = self.AnalyzePopulation(population, iteration)
        print(statistics)


    def AnalyzePopulation(self, population, iteration):
        statistics = {}
        statistics["iteration"] = iteration
        statistics["mean"] = np.mean(population, axis = 0)
        statistics["std"] = np.std(population, axis = 0)

        k, _ = optimalK(population, maxClusters=20)
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


        statistics["cardinality_of_clusters"] = cardinality_of_clusters
        statistics["std_in_clusters"] = std_in_clusters

        return statistics
        