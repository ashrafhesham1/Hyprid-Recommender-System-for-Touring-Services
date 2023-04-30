import sys
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(script_directory,'./../data'))
sys.path.append(os.path.join(script_directory,'./../model'))


from ContentKnn import ContentKnn
from EnsembleAlgorithm import EnsembleAlgorithm
from SVDKNNBaseline import SVDKNNBaseline

from Data import Data

class Model:
    def __init__(self) -> None:
        self.data = Data()
        self.data.load()
        ratings_df = self.data.get_ratings_df()
        items = self.data.get_items()

        self.model = EnsembleAlgorithm([SVDKNNBaseline(ratings_df), ContentKnn(items)])

    def fit(self):
        '''
        fit the model to the trainset
        '''
        self.model.fit(self.data.get_ratings().build_full_trainset())
    
    def predict(self, user, item_id, k = 200):
        '''
        given user and item it predict the rating that the user may give to the item
        -user may be id of user existing in the trainset or dictionary of new user ratings
        Args:
            user: (int|str) id of user in the trainset || (dict) of new user maps items to ratings
            item_id: (int|str) id of the item
            k: (int) number of neighbors
        '''
        if isinstance(user, int) or isinstance(user, str):
            return self._clip(self.model.predict(str(user), str(item_id)).est)

        if isinstance(user, dict):
            return self._clip(self.model.predict_new_user(user, str(item_id), k = k)[0])

        raise Exception('user data must be id for old user (int|str) or dictionary for new user(dict)')        
    
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