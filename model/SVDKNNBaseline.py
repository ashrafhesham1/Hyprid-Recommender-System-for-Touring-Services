import numpy as np
from sklearn.metrics import pairwise_distances
from surprise import SVDpp
import pandas as pd
from surprise import AlgoBase

class SVDKNNBaseline(AlgoBase):
    
    def __init__(self, ratings_df):
        AlgoBase.__init__(self)
        self.ratings = None
        self.SVDModel = SVDpp()
        self.ratings_df = ratings_df

    def fit(self, trainset):
        '''
         given a trainset it fits the model to it
         Args:
            trainset: (surprise.Trainset) of ratings
        Returns:
         '''
        AlgoBase.fit(self, trainset)
        self.SVDModel.fit(trainset)
        self._generate_ratings_matrix(self.ratings_df)
        self._generate_biases()

    def _generate_ratings_matrix(self, ratins_df):
        '''
        given a Dataframe of ratings it convert to matrix of (users,items)
        Args:
            ratins_df: (np.array) of tuples of the form (userId, itemId, rating)
        Returns:
        '''
        self.all_items = ratins_df['item'].unique()

        self.users_ratings_matrix = pd.DataFrame(columns=self.all_items)
        for r in ratins_df.iterrows():
            self.users_ratings_matrix.loc[r[1]['user'],r[1]['item']] = r[1]['rating']

        self.users_ratings_matrix = self.users_ratings_matrix.fillna(0)
        
        self.similarity_idx_to_id={}
        for idx, id_ in enumerate(self.users_ratings_matrix.index):
            self.similarity_idx_to_id[idx] = id_
    
    def _generate_user_ratings_vector(self, user_ratings_dict):
        '''
        given a users ratings as dictionary it convert it into vector with a length of the items
        Args:
            user_ratings_dict: (dict) that maps items to ratings
        Returns:
            user_ratings_vector: (np.array) with a length of the items
        '''
        user_ratings_vector = pd.DataFrame(columns=self.all_items)
        for i in self.all_items:
            user_ratings_vector.loc['new',i] = user_ratings_dict[i] if i in user_ratings_dict.keys() else 0

        return user_ratings_vector
    
    def get_neighbors_data(self, user_ratings_vector, k = 40):
        '''
        given a users ratings it gets top k nearest neighbor to the user 
        Args:
            user_ratings_vector: (np.array) with a length of the items
            k: (int) number of neighbors
        Returns:
            top_k_neighbors: (list) of tuples of the form (item id, similarity, bias)  
        '''
        # measuring users similarities and getting top similar users
        user_similarities = 1 - pairwise_distances(self.users_ratings_matrix, user_ratings_vector, metric='cosine')
        top_k_neighbors = sorted(enumerate(user_similarities), key=lambda x:x[1][0], reverse=True)[:k]
        top_k_neighbors = [(int(self.similarity_idx_to_id[x]), y[0], self.users_bias[x]) for x,y in top_k_neighbors]
        return top_k_neighbors
    
    def _generate_biases(self):
        '''
        uses the train data to generate the bias from the baselin for each user in the trainset
        '''
        # Calculate the mean rating of all users and items
        self.global_mean = np.nanmean( self.users_ratings_matrix.replace(0, np.NaN).values)
        users_means = np.array(np.nanmean( self.users_ratings_matrix.replace(0, np.NaN), axis=1))
        items_means = np.array(np.nanmean( self.users_ratings_matrix.replace(0, np.NaN), axis=0))

        # Calculate the user bias and item bias
        self.users_bias = users_means - self.global_mean
    
    def estimate(self, u, i):
        '''
        given an inner user id  and inner item id it predict the rating the user may give to the item
        Args:
            u: (int) user inner id
            i: (int) item inner id
        Returns:
            prediction: (float) users predicted rating to the item
        '''
        return self.SVDModel.estimate(u, i)
    
    def predict(self, u, i):
        '''
        given a user raw id and item it predict the ratings that the user may give to the item
        Args:
            u: (int|str) user id
            item_id: (int|dtr) the id of the new item
        Returns:
            prediction: (float) users predicted rating to the item
        '''
        return self._clip(self.SVDModel.predict(str(u), str(i)).est)
    
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
        user_ratings_vector = self._generate_user_ratings_vector(user_ratings_dict)
        k_neighbors = self.get_neighbors_data(user_ratings_vector, k = k)

        user_bias = np.array(np.nanmean(user_ratings_vector.replace(0, np.NaN), axis=1)) - self.global_mean

        # generate prediction as average user ratings weighted by similarity scores
        weighted_sum, sim_sum, bias_sum = 0, 0, 0
        for id, sim, bias in k_neighbors:
            rating = self.predict(str(id), str(item_id))
            weighted_sum += sim * rating
            sim_sum += sim
            bias_sum += bias
        
        bias_difference = user_bias - (bias_sum / k)
        neighbors_prediction = weighted_sum / sim_sum
        
        prediction = neighbors_prediction + (((bias_difference / 5) * neighbors_prediction))
        return self._clip(prediction[0])
    
    def _clip(self,val, lower = 0.0, upper = 5.0):
        '''
        given a number , upper bound, and lower bound it clip the number to be between the two bounds
        Args:
            val: (float) the number to be clipped
            lower: (float) the upper bound
            upper: (float) the lower bound
        Returns:
            clipped_val: (float) the number clipped
        '''
        clipped_val = min(val, upper)
        clipped_val = max(clipped_val,lower)

        return clipped_val