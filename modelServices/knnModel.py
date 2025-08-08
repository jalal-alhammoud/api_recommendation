
from surprise import  Reader,Dataset 
import pandas as pd
from surprise import KNNBasic 
from surprise.model_selection import cross_validate
import pickle
import logging
import os
from config.config import Config


class KnnModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data = self.load_and_preprocess_data()
        self.model = None
        self.load_model()

    def load_and_preprocess_data(self):
            interactions = pd.read_csv(Config.DATA_PATHS['reviews']).dropna()
            return {
                'interactions': interactions,
            }
    
    def train_knn(self,similarity='cosine' ,user_based=True, k=30):
        try: 
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(self.data['interactions'][['user_id', 'product_id', 'rating']], reader)
            # Create a user-based collaborative filtering model 
            sim_options = { 'name': similarity, 'user_based': user_based ,'k': k,'min_k': 5 
            }
            # Initialize the KNNBasic algorithm 
            self.model = KNNBasic(sim_options=sim_options)
            # Perform cross-validation 
            cross_validate(self.model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
            trainset = data.build_full_trainset()
            self.model.fit(trainset)
            with open(Config.DATA_PATHS['knn_model'], 'wb') as f:
                    pickle.dump(self.model, f)
        except Exception as e:
            self.logger.error(f"خطأ في تدريب SVD: {e}")
            raise

    def load_model(self):
        try:
            # تحميل نموذج Knn
            if os.path.exists(Config.DATA_PATHS['knn_model']):
                with open(Config.DATA_PATHS['knn_model'], 'rb') as f:
                    self.model = pickle.load(f)
        
        except Exception as e:
            self.logger.error(f"خطأ في تحميل النماذج: {e}")

    def recommend_with_knn(self, user_id, product_id):
        try:
            knn_pred = self.model.predict(user_id, product_id).est 
            return knn_pred      
        except KeyError:                    
            knn_pred=2.5  # قيمة افتراضية إذا لم يوجد التفاعل
            return knn_pred
        except Exception as e:
            print(f"خطأ في التنبؤ: {e}")
            raise         