
from PerformanceMetrics import PerformanceMetrics
from EvaluationData import EvaluationData

class EvaluatedAlgorithm:
    
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
        
    def Evaluate(self, data_evaluation, doTopN, n=10):

        self.algorithm.fit(data_evaluation.GetTrainSet())
        predictions = self.algorithm.test(data_evaluation.GetTestSet())

        metrics_dict = {}

        metrics_dict["RMSE"] = PerformanceMetrics.RMSE(predictions)
        metrics_dict["MAE"] = PerformanceMetrics.MAE(predictions)
        
        if (doTopN):
            
            self.algorithm.fit(data_evaluation.GetLOOCVTrainSet())

            leftOutPredictions = self.algorithm.test(data_evaluation.GetLOOCVTestSet())        
            allPredictions = self.algorithm.test(data_evaluation.GetLOOCVAntiTestSet())
            topNPredicted = PerformanceMetrics.ComputeTopN(allPredictions, n)

           
            metrics_dict["HR"] = PerformanceMetrics.HitRate(topNPredicted, leftOutPredictions)   
            metrics_dict["cHR"] = PerformanceMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions)
            
            self.algorithm.fit(data_evaluation.GetFullTrainSet())
            allPredictions = self.algorithm.test(data_evaluation.GetFullAntiTestSet())
            topNPredicted = PerformanceMetrics.ComputeTopN(allPredictions, n)

                
            metrics_dict["Coverage"] = PerformanceMetrics.UserCoverage(  topNPredicted, 
                                                                   data_evaluation.GetFullTrainSet().n_users, 
                                                                   ratingThreshold=4.0)

            metrics_dict["Diversity"] = PerformanceMetrics.Diversity(topNPredicted, data_evaluation.GetSimilarities())

            metrics_dict["Novelty"] = PerformanceMetrics.Novelty(topNPredicted, 
                                                            data_evaluation.GetPopularityRankings())
        
        return metrics_dict
    
    def GetName(self):
        return self.name
    
    def GetAlgorithm(self):
        return self.algorithm
    
    