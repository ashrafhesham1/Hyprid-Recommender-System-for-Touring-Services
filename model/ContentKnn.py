from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class ContentKnn(AlgoBase):
    def __init__(self, items, k = 40):
        AlgoBase.__init__(self)
        self.items = items
        self.k = k
        self.similarityMatrix = None

        self._generate_item_similarity_matrix_index_mapping()

    def fit(self, trainset):
        '''
         given a trainset it fits the model to it
         Args:
            trainset: (surprise.Trainset) of ratings
        Returns:
         '''
        AlgoBase.fit(self, trainset)
        self._compute_similarity_matrix()
    
    def _compute_similarity_matrix(self):
        '''
        preprocess the items and compute the similarities among them then store it in self.similarityMatrix
        '''
        items = self.items

        # encoding categorical data
        subcategories = [cats[0].split('|') + cats[1].split('|') for cats in zip(items['subcategory'],items['subtype'])]

        mlb = MultiLabelBinarizer()
        catsOneHots = mlb.fit_transform(subcategories)

        # encoding textual data
        nameDesc = pd.Series(items['name'] +  items['description'])

        vectorizer = TfidfVectorizer()
        textTfidf = vectorizer.fit_transform(nameDesc)

        #normalize numerical features
        ratingsNormalizer = MinMaxScaler()
        normalizedRatings = ratingsNormalizer.fit_transform(np.array(items['rating']).reshape(-1, 1))


        revNormalizer = MinMaxScaler()
        normalizedRev = revNormalizer.fit_transform(np.array(items['num_reviews']).reshape(-1, 1))

        numericalFeatures = np.hstack((normalizedRatings, normalizedRev))

        #concatenate data
        all_features = hstack((catsOneHots, textTfidf, numericalFeatures))  

        self.similarityMatrix = cosine_similarity(all_features)

    def _generate_item_similarity_matrix_index_mapping(self):
        '''
        generate mapping between items raw ids and the index of the item in the similarity matrix
        '''
        self.item_id_to_similarity_idx = {}
        self.similarity_idx_to_item_id = {}
        for inner_idx, item_id in enumerate(self.items.index):
            self.item_id_to_similarity_idx[item_id] = inner_idx
            self.similarity_idx_to_item_id[inner_idx] = item_id
    
    def _get_similarity(self, item1_id, item2_id):
        '''
        given two items raw ids it gets the similarity between them from the similarity matrix
        Args:
            item1_id: (int) raw if of the first item
            item2_id: (int) raw if of the second item
        Returns:
            similarity: (float) similarity between the two items
        '''
        item1_sim_idx = self.item_id_to_similarity_idx[int(item1_id)]
        item2_sim_idx = self.item_id_to_similarity_idx[int(item2_id)]

        return self.similarityMatrix[item1_sim_idx, item2_sim_idx]
    
    def get_neighbors(self,u, i, k = 40, user_ratings_dict = {}):
        '''
        given a user (unknown / known) and a known item it gets the most similar items from the items
        that the user has rate to the given item
        Args:
            u: (int) user inner id - only for known users
            i: (int) item inner id for known users - item raw id for unknow users
            k: (int) number of neighbors to consider while measuring similarity
            user_ratings_dice: (dict) of user ratings maps item id to rating - only for unknown users
        Returns:
            neighbors: (list) of tuples (similarity, rating) for the top k similar neighbors sorted by similarity
        '''

        neighbors = []

        # generate similarity between the item and everyitem the user rated
        # similarity for known user 
        if self.trainset.knows_user(u):
            item_raw_id = self.trainset.to_raw_iid(i)
            for rating in self.trainset.ur[u]:
                cur_item_raw_id = self.trainset.to_raw_iid(rating[0])
                similarity = self._get_similarity(item_raw_id ,cur_item_raw_id)
                neighbors.append((similarity, rating[1]))
        
        #similarity for unknown user
        else:
            item_raw_id = str(i)
            for rating in user_ratings_dict.items():
                    cur_item_raw_id = str(rating[0])
                    similarity = self._get_similarity(item_raw_id ,cur_item_raw_id)
                    neighbors.append((similarity, rating[1]))        
        
        #computing top k similar neighbors
        k = min(k, len(neighbors))
        k_neighbors = sorted(neighbors, key=lambda x:x[0], reverse=True)[:k]

        return(k_neighbors)

    def estimate(self, user_id, item_id):
        '''
        given an inner user id  and inner item id it predict the rating the user may give to the item
        Args:
            user_id: (int) user inner id
            item_id: (int) item inner id
        Returns:
            prediction: (float) users predicted rating to the item
        '''
        if not (self.trainset.knows_user(user_id) and self.trainset.knows_item(item_id)):
            raise PredictionImpossible('User and/or item is unkown.')
    
        k_neighbors = self.get_neighbors(user_id, item_id, k = self.k)

        # generate prediction as average user ratings weighted by similarity scores
        weighted_sum, sim_sum = 0, 0
        for sim, rating in k_neighbors:
            weighted_sum += sim * rating
            sim_sum += sim
        
        return self._clip(weighted_sum / (sim_sum + .00001))
    
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
        k_neighbors = self.get_neighbors('new', item_id, k = k, user_ratings_dict=user_ratings_dict)

        # generate prediction as average user ratings weighted by similarity scores
        weighted_sum, sim_sum = 0, 0
        for sim, rating in k_neighbors:
            weighted_sum += sim * rating
            sim_sum += sim
        
        return self._clip(weighted_sum / (sim_sum+.00001))

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