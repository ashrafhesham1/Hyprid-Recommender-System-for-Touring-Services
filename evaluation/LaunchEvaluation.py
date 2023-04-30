import sys
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

#sys.path.append(os.path.abspath('./data'))
sys.path.append(os.path.join(script_directory,'./../data'))
sys.path.append(os.path.join(script_directory,'./../model'))

from Data import Data
from Evaluator import Evaluator
from surprise import KNNWithMeans, NormalPredictor, SVDpp
from ContentKnn import ContentKnn
from EnsembleAlgorithm import EnsembleAlgorithm
from SVDKNNBaseline import SVDKNNBaseline
from Model import Model

import random
import numpy as np
import numpy as np

def load_data():
    dataset = Data()
    print("Loading ratings...")
    dataset.load()
    data = dataset.get_ratings()
    print("\nComputing popularity ranks so we can measure novelty later...")
    rankings = dataset.get_popularity_ranking()
    items = dataset.get_items()

    return (data, rankings, items)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(evaluationData, rankings, items) = load_data()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)
#ES 

#content_based 
content = ContentKnn(items)
evaluator.add_algoritm(content, 'content')

#items-Knn 
IKNN = KNNWithMeans(sim_options={'name':'cosine', 'user_based':False})
evaluator.add_algoritm(IKNN, 'IKNN')

#SVD
svd = SVDpp()
evaluator.add_algoritm(svd, 'SVD')

#ES 
ens = EnsembleAlgorithm([KNNWithMeans(sim_options={'name':'cosine', 'user_based':False}), ContentKnn(items)])
evaluator.add_algoritm(ens, 'KNN-IC')


#ES 
ens2 = EnsembleAlgorithm([KNNWithMeans(sim_options={'name':'cosine', 'user_based':True}), ContentKnn(items)])
evaluator.add_algoritm(ens2, 'KNN-UC')

#ES 
ens3 = EnsembleAlgorithm([KNNWithMeans(sim_options={'name':'cosine', 'user_based':True}), KNNWithMeans(sim_options={'name':'cosine', 'user_based':False}), ContentKnn(items)])
evaluator.add_algoritm(ens3, 'KNN-IUC')

#ES 
ens4 = EnsembleAlgorithm([SVDpp(), ContentKnn(items)])
evaluator.add_algoritm(ens4, 'SVD-KNNC')

# Just make random recommendations
Random = NormalPredictor()
evaluator.add_algoritm(Random, "Random")


# Fight!
evaluator.evaluate(True)