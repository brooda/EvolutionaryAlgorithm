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
        
        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return resultsdf #(gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

class Analyzer:
    def __init__(self, log_dir):
        self.log_dir = log_dir

        print(self.log_dir)

        try:
            os.mkdir(self.log_dir)
        except OSError:
            print ("Creation of the directory %s failed" % self.log_dir)
        else:
            print ("Successfully created the directory %s " % self.log_dir)
            

    def AnalyzePopulationAndSaveToLog(self, population, iteration):
        statistics = self.AnalyzePopulation(population, iteration)
    
        print(statistics)

        with open(os.path.join(self.log_dir, f'{iteration}.json'), 'wb') as outfile:
            pickle.dump(statistics, outfile)


    def AnalyzePopulation(self, population, iteration):
        statistics = {}
        statistics["iteration"] = iteration
        statistics["mean"] = np.mean(population, axis = 0).tolist()
        statistics["std"] = np.std(population, axis = 0).tolist()

        k, _ = optimalK(population, maxClusters=20)
        statistics["optimal_k"] = k


        kmeans = KMeans(n_clusters=k)
        kmeans.fit(population)
        
        statistics["cluster_centers"] = kmeans.cluster_centers_.tolist()

        ### https://stackoverflow.com/a/50804098/4793865
        cluster_to_representants = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

        cardinality_of_clusters = {}
        std_in_clusters = {}

        for k, v in cluster_to_representants.items():
            cardinality_of_clusters[k] = v.size
            std_in_clusters[k] = np.std(population[v], axis = 0).tolist()


        statistics["cardinality_of_clusters"] = cardinality_of_clusters
        statistics["std_in_clusters"] = std_in_clusters

        return statistics
        