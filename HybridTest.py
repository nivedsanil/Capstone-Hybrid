
import random
from ContentFiltering import ContentFiltering
import numpy as np
from surprise import SVD
from HybridAlgorithm import HybridAlgorithm
from surprise import KNNBasic
from Evaluator import Evaluator
from MovieLensData import MovieLensData

np.random.seed(0)
random.seed(0)

def LoadData():
    movielens = MovieLens()
    print("Loading all ratings and computing popularity ranks from Movie Lens...")
    data = movielens.loadMovieLens()
    ranks= movielens.computePopularity()
    return (movielens, data, ranks)

(movielens, data_evaluation, ranks) = LoadData()

evaluator = Evaluator(data_evaluation, ranks)

print("Computing content-based similarity based on Genre, Year and Mise En Scene similarity")
ContentBased = ContentFiltering()

print("Computing Item-Based Collaborative Filtering")
itemKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': False}, k=15) 

Hybrid = HybridAlgorithm([ContentBased,itemKNN], [0.75,0.25])

evaluator.AddAlgorithm(ContentBased, "Content Based Filtering")
evaluator.AddAlgorithm(itemKNN, "Item-Based Collaborative Filtering")
evaluator.AddAlgorithm(Hybrid, "Hybrid")

evaluator.Evaluate(True)

evaluator.SampleTopNRecs(movielens)



