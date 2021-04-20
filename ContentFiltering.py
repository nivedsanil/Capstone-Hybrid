from surprise import AlgoBase
from surprise import PredictionImpossible
from MovieLensData import MovieLensData
import math
import numpy as np
import heapq

class ContentFiltering(AlgoBase):

    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        ml = MovieLensData()
        genres = ml.returnGenres()
        years = ml.returnYears()
        mes = ml.returnMES()
            
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))

        for thisRating in range(self.trainset.n_items):
 
            for otherRating in range(thisRating+1, self.trainset.n_items):
                thisMovieID = int(self.trainset.to_raw_iid(thisRating))
                otherMovieID = int(self.trainset.to_raw_iid(otherRating))
                # print("Computing Genre Similarity")
                genreSimilarity = self.computeGenreSimilarity(thisMovieID, otherMovieID, genres)
                # print("Computing Year Similarity")
                yearSimilarity = self.computeYearSimilarity(thisMovieID, otherMovieID, years)
                # print("Computing Mise En Scene Similarity")
                mesSimilarity = self.computeMiseEnSceneSimilarity(thisMovieID, otherMovieID, mes)
                self.similarities[thisRating, otherRating] = genreSimilarity * yearSimilarity*mesSimilarity
                self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]

                
        return self
    
    def computeGenreSimilarity(self, movie1, movie2, genres):
        
        genres1 = genres[movie1]
        genres2 = genres[movie2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        
        return sumxy/math.sqrt(sumxx*sumyy)
    
    def computeYearSimilarity(self, movie1, movie2, years):
    
        diff = abs(years[movie1] - years[movie2])
        sim = math.exp(-diff / 10.0)
        return sim
    
    def computeMiseEnSceneSimilarity(self, movie1, movie2, mesData):
        
        mv1 = mesData[movie1]
        mv2 = mesData[movie2]
        if (mv1 and mv2):
            lighting = math.fabs(mv1[2] - mv2[2])
            motion = math.fabs(mv1[0] - mv2[0])
            shotLength = math.fabs(mv1[4] - mv2[4])
            numShots = math.fabs(mv1[1] - mv2[1])
            colorVariance = math.fabs(mv1[3] - mv2[3])

            return  lighting * motion * colorVariance * numShots * shotLength 
        else:
            return 0

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            genreSimilarity = self.similarities[i,rating[0]]
            neighbors.append( (genreSimilarity, rating[1]) )
        
        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        
        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating
            
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating
    