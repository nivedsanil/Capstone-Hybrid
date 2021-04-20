
from surprise import SVD
from surprise import KNNBasic
from MovieLens import MovieLens
from ContentKNNAlgorithm import ContentKNNAlgorithm
from HybridAlgorithm import HybridAlgorithm
from Evaluator import Evaluator

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)


#Content Based Algorithm
print("Computing content-based similarity based on Genre, Year and Mise En Scene similarity")
ContentKNN = ContentKNNAlgorithm()

itemKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': False}, k=15) 

#Combine them
Hybrid = HybridAlgorithm([ContentKNN,itemKNN], [0.75,0.25])

evaluator.AddAlgorithm(ContentKNN, "Content Based Filtering")
evaluator.AddAlgorithm(itemKNN, "Item-Based Collaborative Filtering")
evaluator.AddAlgorithm(Hybrid, "Hybrid")

# Fight!
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
