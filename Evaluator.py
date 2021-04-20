
from EvaluationData import EvaluationData
from EvaluatedAlgorithm import EvaluatedAlgorithm
from MovieLens import MovieLens

class Evaluator:
    
    algorithms = []
    
    def __init__(self, dataset, rankings):
        ed = EvaluationData(dataset, rankings)
        self.dataset = ed
        
    def AddAlgorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)
        
    def Evaluate(self, doTopN):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.GetName(), "...")
            results[algorithm.GetName()] = algorithm.Evaluate(self.dataset, doTopN)

        # Print results
        print("\n")
        
        if (doTopN):
            print("{:<50} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty"))
            for (name, metrics) in results.items():
                print("{:<50} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                                      metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))
                
        
    def SampleTopNRecs(self, ml, testSubject=85, k=10):
        
        for algo in self.algorithms:
            print("\nUsing recommender ", algo.GetName())
            
            print("\nBuilding recommendation model...")
            trainSet = self.dataset.GetFullTrainSet()
            algo.GetAlgorithm().fit(trainSet)
            
            print("Generating recommendations...")
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
        
            predictions = algo.GetAlgorithm().test(testSet)
            # print(predictions)
            
            recommendations = []
            
            print ("\nRecommendation Generated:")
            
            years = ml.getYears()

            for userID, movieID, actualRating, estimatedRating, _ in predictions:

                intMovieID = int(movieID)

                if(years[intMovieID] > 2000):
                    recommendations.append((intMovieID, estimatedRating))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            for ratings in recommendations[:10]:
                print(ml.getMovieName(ratings[0]), ratings[1])
                

            
            
    
    