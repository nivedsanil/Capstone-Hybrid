
import math
import numpy as np
import heapq
from MovieLensData import MovieLensData
from surprise import AlgoBase

class ContentFiltering(AlgoBase):

    def __init__(self, k=60, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainData):

        movielens = MovieLens()
        AlgoBase.fit(self, trainData)

        mesData = movielens.returnMES()
        yearsData = movielens.returnYears()
        genreData = movielens.returnGenres()
        
        self.similarities = np.zeros((self.trainData.n_items, self.trainData.n_items))

        for thisMovieRating in range(self.trainData.n_items):

            for otherMovieRating in range(thisMovieRating+1, self.trainData.n_items):
                this_ID = int(self.trainData.to_raw_iid(thisMovieRating))
                other_ID = int(self.trainData.to_raw_iid(otherMovieRating))

                calcMesSim = self.calcMesSim(this_ID, other_ID, mesData)
                calcGenreSim = self.calcGenreSim(this_ID, other_ID, genreData)
                calcYearSim = self.calcYearSim(this_ID, other_ID, yearsData)
                
                
                self.similarities[thisMovieRating, otherMovieRating] = calcGenreSim*calcYearSim*calcMesSim
                self.similarities[otherMovieRating, thisMovieRating] = self.similarities[thisMovieRating, otherMovieRating]
                
        return self

    
    def calcMesSim(self, movie1, movie2, mesData):
        
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

    def calcYearSim(self, movie1, movie2, yearsData):

    diff = abs(yearsData[movie1] - yearsData[movie2])
    sim_score = math.exp(-diff / 10.0)
    return sim_score

    def calcGenreSim(self, movie1, movie2, genreData):

    xx = 0
    yy = 0
    xy = 0
    
    genre_mv1 = genres[movie1]
    genre_mv2 = genres[movie2]

    for i in range(len(genre_mv1)):
        x = genre_mv1[i]
        y = genre_mv2[i]
        xx += x * x
        yy += y * y
        xy += x * y

    cos_sim = xy/math.sqrt(xx*yy)
    
    return cos_sim

    def estimate(self, u, i):

        neighbour_list = []
        for rating in self.trainData.ur[u]:
            calcGenreSim = self.similarities[i,rating[0]]
            neighbour_list.append( (calcGenreSim, rating[1]) )

        k_neighbors = heapq.nlargest(self.k, neighbour_list, key=lambda t: t[0])

        total_simScore = 0
        weighted_totalSimScore = 0

        for (similarityScore, rating) in k_neighbors:
            if (similarityScore > 0):
                total_simScore += similarityScore
                weighted_totalSimScore += simScore * rating
            

        predictedRating = weighted_totalSimScore / total_simScore

        return predictedRating
    