import pandas as pd
from surprise import Dataset, Reader,SVD
import pickle
import logging
import os
from config.config import Config

class SvdModel:
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
    
    def train_svd(self):
        """تدريب نموذج SVD"""
        try:
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(
                self.data['interactions'][['user_id', 'product_id', 'rating']],
                reader
            )
            #  trainset = self.data['interactions'].build_full_trainset()
            self.model = SVD(**Config.MODEL_PARAMS['svd'])
            trainset = data.build_full_trainset()
            self.model.fit(trainset)
            
            # حفظ النموذج
            with open(Config.DATA_PATHS['svd_model'], 'wb') as f:
                pickle.dump(self.model, f)
                
        except Exception as e:
            self.logger.error(f"خطأ في تدريب SVD: {e}")
            raise
    
    def load_model(self):
        try:
            # تحميل نموذج SVD
            if os.path.exists(Config.DATA_PATHS['svd_model']):
                with open(Config.DATA_PATHS['svd_model'], 'rb') as f:
                    self.model = pickle.load(f)
        
        except Exception as e:
            self.logger.error(f"خطأ في تحميل النماذج: {e}")

    def recommend_with_svd(self, user_id, product_id):
        try:
            svd_pred = self.model.predict(user_id, product_id).est 
            return svd_pred      
        except KeyError:                    
            svd_pred=2.5  # قيمة افتراضية إذا لم يوجد التفاعل
            return svd_pred
        except Exception as e:
            print(f"خطأ في التنبؤ: {e}")
            raise         

    