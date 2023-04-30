import numpy as np
from surprise import AlgoBase
from surprise import PredictionImpossible
from sklearn.linear_model import LinearRegression

class EnsembleAlgorithm(AlgoBase):
    def __init__(self, algorithms):
        AlgoBase.__init__(self)
        self.algorithms = algorithms
    
    def fit(self, trainset):
        '''
         Args:
            trainset: (surprise.Trainset) of ratings
        Returns:
         '''
        AlgoBase.fit(self, trainset)

        for i in range(len(self.algorithms)):
            self.algorithms[i].fit(trainset)
        
        x_train, y_train = self._generate_ensemble_train_data(trainset)

        self.model = LinearRegression()
        self.model.fit(x_train,y_train)
        
    def _generate_ensemble_train_data(self, trainset):
        '''
        given a ratings trainsets it generate an ensemble trainset to be used to train the linear regression model
        to generate a final prediction from the submodels predictios
        Args:
            trainset: (surprise.Trainset) of ratings
        Returns:
            x_train: (np.array) of the trainning data
            y_train: (np.array of the targets)
        '''
        x_train = np.zeros((trainset.n_ratings, len(self.algorithms)))
        y_train = np.zeros((trainset.n_ratings))

        for i, (uid, iid, rating) in enumerate(trainset.all_ratings()):
            for j, algorithm in enumerate(self.algorithms):
                algo_prediction = algorithm.estimate(uid, iid)
                x_train[i][j] = algo_prediction if isinstance(algo_prediction, float) else algo_prediction[0]
            
            y_train[i] = rating

        return x_train, y_train
    
    def estimate(self, u, i):
        '''
        given an inner user id  and inner item id it predict the rating the user may give to the item
        Args:
            u: (int) user inner id
            i: (int) item inner id
        Returns:
            prediction: (float) users predicted rating to the item
        '''
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        predictions = []
        for j in range(len(self.algorithms)):
            algo_prediction = self.algorithms[j].estimate(u, i)
            algo_prediction = algo_prediction if isinstance(algo_prediction, float) else algo_prediction[0]
            predictions.append(algo_prediction)

        return self.model.predict(np.array(predictions).reshape(1, -1))[0]

    def predict_new_user(self, user_ratings_dict, item_id, k = 40):
        '''
        given new unknown user ratings and item it predict the ratings that the user may give to the item
        Args:
            user_ratings_dict: (dict) of user ratings maps item id to rating
            item_id: (int) the id of the new item
            k: (int) number of neighbors to be considered while computing rating
        Returns:
            prediction: (float) users predicted rating to the item
        '''
        predictions = []
        for j in range(len(self.algorithms)):
            algo_prediction = self.algorithms[j].predict_new_user(user_ratings_dict, item_id, k = k)
            algo_prediction = algo_prediction if isinstance(algo_prediction, float) else algo_prediction[0]
            predictions.append(algo_prediction)

        return self.model.predict(np.array(predictions).reshape(1, -1))
