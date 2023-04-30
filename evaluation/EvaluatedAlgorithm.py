from RecommenderMetrics import RecommenderMetrics
from EvaluationData import EvaluationData
import gc

class EvaluatedAlgorithm:
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
    
    def evaluate(self, evaluationData, doTopN, n = 10, verbose = True):
        metrics = {}
        recommenderMetrics = RecommenderMetrics()

        #Accuracy
        if (verbose):
            print("Evaluating accuracy...")
        self.algorithm.fit(evaluationData.get_train_set())
        predictions = self.algorithm.test(evaluationData.get_test_set())
        metrics['RMSE'] = recommenderMetrics.RMSE(predictions)
        metrics['MAE'] = recommenderMetrics.MAE(predictions)

        #Top N
        if doTopN:

            # leave one out
            if (verbose):
                print("Evaluating top-N with leave-one-out...")
            self.algorithm.fit(evaluationData.get_LOOCV_train_set())
            leftOutPredictions = self.algorithm.test(evaluationData.get_LOOCV_test_set())

            # predictions for all ratings not in the training set
            allPredictions = self.algorithm.test(evaluationData.get_LOOCV_anti_test_set())

            # Compute top 10 recs for each user
            topNPredicted = recommenderMetrics.get_top_N(allPredictions, n)

            if (verbose):
                print("Computing hit-rate and rank metrics...")
            # See how often we recommended a movie the user actually rated
            metrics["HR"] = recommenderMetrics.get_hit_rate(topNPredicted, leftOutPredictions)   
            # See how often we recommended a movie the user actually liked
            #metrics["cHR"] = recommenderMetrics.get_commulative_hit_rate(topNPredicted, leftOutPredictions)

            gc.collect()
            #Evaluate properties of recommendations on full training set
            '''
            if (verbose):
                print("Computing recommendations with full data set...")
            self.algorithm.fit(evaluationData.get_full_train_set())
            allPredictions = self.algorithm.test(evaluationData.get_full_anti_test_set())
            topNPredicted = recommenderMetrics.get_top_N(allPredictions, n)
            if (verbose):
                print("Analyzing coverage, diversity, and novelty...")

            # Print user coverage with a minimum predicted rating of 4.0:
            metrics["Coverage"] = recommenderMetrics.get_user_coverage(  topNPredicted, 
                                                                   evaluationData.get_full_train_set().n_users, 
                                                                   ratingThreshold=4.0)
            # Measure diversity of recommendations:
            metrics["Diversity"] = recommenderMetrics.get_diversity(topNPredicted, evaluationData.get_similarities())
            
            # Measure novelty (average popularity rank of recommendations):
            metrics["Novelty"] = recommenderMetrics.get_novelty(topNPredicted, evaluationData.get_popularity_rankings())
            '''
        if (verbose):
            print("Analysis complete.")

        return metrics
    
    def get_name(self):
        return self.name
    
    def get_algorithm(self):
        return self.algorithm