import random
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline

class EvaluationData:
    def __init__(self, data, rankings) -> None:
        
        self.rankings = rankings
        self._build_train_test_sets(data)
        self._build_similarit_matrix()
    
    def _build_train_test_sets(self, data):

        self.fullTrainSet = data.build_full_trainset()
        self.fullAntiTestSet = self.fullTrainSet.build_anti_testset()

        # 75/25 train/test split
        self.trainSet, self.testSet = train_test_split(data, test_size=0.25, random_state=40 )

        #subset = random.sample(self.fullTrainSet.all_ratings(), k=1000)

        #leave on out train/test split
        LOOCV = LeaveOneOut(n_splits=1, random_state=40)
        for train, test in LOOCV.split(data):
            self.LOOCVTrain = train
            self.LOOCVTest = test
        
        self.LOOCVAntiTestSet = self.LOOCVTrain.build_anti_testset()
    
    def _build_similarit_matrix(self):
        sim_options = {
            'name': 'cosine',
            'user_based': False
            }
        
        self.simsAlgo = KNNBaseline(sim_options=sim_options)
        self.simsAlgo.fit(self.fullTrainSet)

    def get_full_train_set(self):
        return self.fullTrainSet
    
    def get_full_anti_test_set(self):
        return self.fullAntiTestSet
    
    def get_anti_test_set_for_user(self, testSubject):
        trainset = self.fullTrainSet
        placeHolder = trainset.global_mean 
        antiTestset = []
        u = trainset.to_inner_uid(str(testSubject))
        userItems = set([j for (j,_) in trainset.ur[u]])
        antiTestset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), placeHolder) \
                        for i in trainset.all_items()\
                        if i not in userItems]
        
        return antiTestset


    def get_train_set(self):
        return self.trainSet
    
    def get_test_set(self):
        return self.testSet
    
    def get_LOOCV_train_set(self):
        return self.LOOCVTrain
    
    def get_LOOCV_test_set(self):
        return self.LOOCVTest
    
    def get_LOOCV_anti_test_set(self):
        return self.LOOCVAntiTestSet
    
    def get_similarities(self):
        return self.simsAlgo
    
    def get_popularity_rankings(self):
        return self.rankings