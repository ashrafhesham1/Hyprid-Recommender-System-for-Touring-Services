import itertools

from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:

    def __init__(self) -> None:
        pass

    def MAE(self, predictions):
        return accuracy.mae(predictions, verbose=False)
    
    def RMSE(self, predictions):
        return accuracy.rmse(predictions, verbose=False)
    
    def get_top_N(self, predictions, n = 10, minimumRating = 0.4):
        topN = defaultdict(list)

        for userId, itemId, actualRating, estimsatedRating, _ in predictions:
            if estimsatedRating >= minimumRating:
                topN[int(userId)].append((int(itemId), estimsatedRating))

        for userId, ratings in topN.items():
            ratings.sort(key=lambda x:x[1], reverse=True)
            topN[int(userId)] = ratings[:10]

        return topN               

    def get_hit_rate(self, topNPredictions, leftOutPredictions):
        hits, total = 0, 0
        for leftOut in leftOutPredictions:
            userId, leftOutId = leftOut[0], leftOut[1]
            hit = False
            for itemId, predictions in topNPredictions[int(userId)]:
                if int(itemId) == int(leftOutId):
                    hit = True
                    break
            
            if(hit):
                hits +=1
            
            total +=1
        
        return hits / total
    
    def get_commulative_hit_rate(self, topNPredictions, leftOutPredictions, ratingCutOff = 0):
        hits, total = 0, 0
        for leftOut in leftOutPredictions:
            userId, leftOutId, actualRating = leftOut[0], leftOut[1], leftOut[2]
            if actualRating >= ratingCutOff:
                hit = False
                for itemId, predictions in topNPredictions[int(userId)]:
                    if int(itemId) == int(leftOutId):
                        hit = True
                        break
                
                if(hit):
                    hits +=1
                
                total +=1
        
        return hits / total
    
    def get_rating_hit_rate(self, topNPredictions, leftOutPredictions, printRes = False):
        hits, total, ratingHits = defaultdict(float), defaultdict(float), defaultdict(float)
        for leftOut in leftOutPredictions:
            userId, leftOutId, actualRating = leftOut[0], leftOut[1], leftOut[2]
            hit = False
            for itemId, predictions in topNPredictions[int(userId)]:
                if int(itemId) == int(leftOutId):
                    hit = True
                    break

                if hit:
                    hits[actualRating] += 1
                
                total[actualRating] += 1
        
        for rating in sorted(hits.keys):
            ratingHitRate = hits[rating] / total[rating]
            ratingHits[rating] = ratingHitRate

            if printRes:
                print(rating, ratingHitRate)
        
        return ratingHits
    
    def get_user_coverage(self, topNPredicted, numUsers, ratingThreshold = 0):
        hits = 0
        for userId in topNPredicted.keys():
            hit = False
            for itemId, predictedRating in topNPredicted[userId]:
                if predictedRating >= ratingThreshold:
                    hit = True
                    break

                if hit:
                    hits += 1
        
        return hits / numUsers
    
    def get_diversity(self, topNPredictions, simsAlgo):
        n, total = 0, 0
        simsMatrix = simsAlgo.compute_similarities()
        for userId in topNPredictions.keys():
            pairs = itertools.combinations(topNPredictions[userId],2)
            for pair in pairs:
                item1, item2 = pair[0][0], pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(item1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(item2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1
        
        s = total / n
        return 1 - s
    
    def get_novelty(self, topNPredictions, rankings):
        n, total = 0, 0
        for userId in topNPredictions.keys():
            for rating in topNPredictions[userId]:
                itemId = rating[0]
                rank = rankings[itemId]
                total += rank
                n += 1
        
        return total / n