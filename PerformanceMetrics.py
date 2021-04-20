from surprise import accuracy
import itertools
from collections import defaultdict

class PerformanceMetrics:

    def ComputeTopN(predicted_data, n=10, min_rating=4.0):
        topN_dict = defaultdict(list)


        for user_ID, movie_ID, realRating, predictedRating, _ in predicted_data:
            if (predictedRating >= min_rating):
                topN_dict[int(user_ID)].append((int(movie_ID), predictedRating))

        for user_ID, ratings in topN_dict.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN_dict[int(user_ID)] = ratings[:n]

        return topN_dict

    def RMSE(predicted_data):
        return accuracy.rmse(predicted_data, verbose=False)

    def MAE(predicted_data):
        return accuracy.mae(predicted_data, verbose=False)

  
    def HitRate(predicted_topN, predictions_leftOut):
        total, hits = 0, 0
        for leftOutMovie in predictions_leftOut:

            hit = False
            user_ID = leftOutMovie[0]
            leftOutMovieID = leftOutMovie[1]

            for movie_ID, predictedRating in predicted_topN[int(user_ID)]:
                if (int(leftOutMovieID) == int(movie_ID)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        return hits/total

    def CumulativeHitRate(predicted_topN, predictions_leftOut, ratingCutoff=0):
       
        hits, total= 0, 0
        for user_ID, leftOutMovieID, realRating, predictedRating, _ in predictions_leftOut:

            if (realRating >= ratingCutoff):
                hit = False
                for movie_ID, predictedRating in predicted_topN[int(user_ID)]:
                    if (int(leftOutMovieID) == movie_ID):
                        hit = True
                        break
                if (hit) :
                    hits += 1
                total += 1

        return hits/total

    def UserCoverage(predicted_topN, total_users, ratingThreshold=0):
        hits = 0
        for user_ID in predicted_topN.keys():
            hit = False
            for movie_ID, predictedRating in predicted_topN[user_ID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / total_users

    def Diversity(predicted_topN, sim_algo):
        
        similarity_matrix = sim_algo.compute_similarities()
        
        n, total = 0, 0
        
        for user_ID in predicted_topN.keys():
            pairs = itertools.combinations(predicted_topN[user_ID], 2)
            for pair in pairs:
                mv1 = pair[0][0]
                mv2 = pair[1][0]
                innerID1 = sim_algo.trainset.to_inner_iid(str(mv1))
                innerID2 = sim_algo.trainset.to_inner_iid(str(mv2))
                similarity = similarity_matrix[innerID1][innerID2]
                total += similarity
                n += 1

        Similarities = total / n
        return (1-Similarities)

    def Novelty(predicted_topN, rankings):
        n,total = 0, 0
        for user_ID in predicted_topN.keys():
            for rating in predicted_topN[user_ID]:
                movie_ID = rating[0]
                rank = rankings[movie_ID]
                total += rank
                n += 1
        return total / n
