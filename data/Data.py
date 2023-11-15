import sys
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(script_directory,'./..'))

import csv
import pandas as pd 

from surprise import Dataset
from surprise import Reader

from utils import Env
from collections import defaultdict

import os

class Data:
    def __init__(self):
        env = Env()

        self.items_path = env.get('items_path')
        self.ratings_path = env.get('ratings_path')

        self.itemId_to_name = {}
        self.itemName_to_id = {}

        self.ratings_dataset = None

        self.loaded = False
    
    def load(self):
        # generate the dataset
        reader = Reader(line_format = 'user item rating timestamp', sep = ',', skip_lines = 1)
        self.ratings_dataset = Dataset.load_from_file(self.ratings_path, reader)

        d_df = pd.read_csv(self.ratings_path)


        self._generate_name_id_mapping()
        self._load_items_features()
        self._load_ratings_df()
        self.loaded = True
    
    def _generate_name_id_mapping(self):
        with open(self.items_path, newline='', encoding='UTF-8') as items:
            itemsReader = csv.reader(items)
            next(itemsReader)
            for record in itemsReader:
                id, name = record[1], record[2]
                self.itemId_to_name[id] = name
                self.itemName_to_id[name] = id

    def _load_items_features(self):
        self.items = pd.read_csv(self.items_path, index_col=0)
        self.itemsSubcategories = list(set(cat for item in self.items['subcategory'].tolist() for cat in item.split('|')))
        self.itemsSubtypes = list(set(cat for item in self.items['subtype'].tolist() for cat in item.split('|')))
        
    def _load_ratings_df(self):
        self.ratings_dataframe = pd.read_csv(self.ratings_path)
        self.ratings_dataframe = self.ratings_dataframe.drop(['timestamp'], axis=1)
        
    def get_ratings(self):
        if not self.loaded:
            return None
        
        return self.ratings_dataset
    
    def get_item_id(self, name):
        if not self.loaded:
            return None
        
        return self.itemName_to_id[name]
    
    def get_item_name(self, id):
        if not self.loaded:
            return None
        
        return self.itemId_to_name[id]

    def get_user_ratings(self, userId):
        userRatings = []
        with open(self.ratings_path, newline='', encoding='UTF-8') as ratings:
            ratingsReader = csv.reader(ratings)
            next(ratingsReader)
            for record in ratingsReader:
                curUser = record[0]
                if curUser != userId:
                    continue
                item, rating = record[1], record[2]
                userRatings.append((item, rating))  

        return userRatings
    
    def get_popularity_ranking(self):
        popularity, rankings = defaultdict(int), defaultdict(int)
        with open(self.ratings_path, newline='', encoding='UTF-8') as ratings:
            ratingsReader = csv.reader(ratings)
            next(ratingsReader)
            for record in ratingsReader:
                item = record[1]
                popularity[item] += 1
        
        rank = 1
        for item, _ in sorted(popularity.items(), key=lambda x:x[1], reverse=True):
            rankings[item] = rank
            rank += 1
        
        return rankings

    def get_items_subcategories(self):
        return self.itemsSubcategories
    
    def get_items_subtypes(self):
        return self.itemsSubtypes
   
    def get_ratings_df(self):
        return self.ratings_dataframe
  
    def get_items(self):
        return self.items