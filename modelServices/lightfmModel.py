import pandas as pd
import numpy as np
from surprise.model_selection import train_test_split
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightfm.data import Dataset as dtset
from lightfm import LightFM
import pickle
from config.config import Config
import os

class LightfmModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data = self.load_and_preprocess_data()
        self.model = None
        self.load_model()


    def load_and_preprocess_data(self):
            # Load data
            users = pd.read_csv(Config.DATA_PATHS['users'])
            products = pd.read_csv(Config.DATA_PATHS['products'])
            interactions = pd.read_csv(Config.DATA_PATHS['reviews']).dropna()

            # Ensure we only keep users and products that exist in interactions
            interactions = interactions[
                interactions['user_id'].isin(users['user_id']) &
                interactions['product_id'].isin(products['product_id'])
            ]

            # Get unique user and product IDs that have interactions
            active_users = interactions['user_id'].unique()
            active_products = interactions['product_id'].unique()

            # Filter datasets
            users = users[users['user_id'].isin(active_users)]
            products = products[products['product_id'].isin(active_products)]
            sensitive_features = users[['user_id', 'gender']].copy()

            # Process numerical features
            user_scaler = StandardScaler()
            users[['age']] = user_scaler.fit_transform(users[['age']])

            product_scaler = StandardScaler()
            products['price'] = product_scaler.fit_transform(products[['price']])

            # Process categorical features
            user_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            user_categorical = user_encoder.fit_transform(users[['gender', 'location']])

            product_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            product_categorical = product_encoder.fit_transform(products[['category', 'brand']])

            # Create mapping dictionaries for user and product indices
            user_id_to_idx = {user_id: idx for idx, user_id in enumerate(users['user_id'])}
            product_id_to_idx = {product_id: idx for idx, product_id in enumerate(products['product_id'])}

            # Create interaction matrix
            interaction_matrix = np.zeros((len(users), len(products)))
            for _, row in interactions.iterrows():
                user_idx = user_id_to_idx[row['user_id']]
                product_idx = product_id_to_idx[row['product_id']]
                interaction_matrix[user_idx, product_idx] = row['rating']

            return {
                'users': users,
                'products': products,
                'interactions': interactions,
                'interaction_matrix': interaction_matrix,
                'sensitive_features': sensitive_features,
                'features': {
                    'user_numerical': users[['age']].values,
                    'user_categorical': user_categorical,
                    'product_numerical': products[['price']].values,
                    'product_categorical': product_categorical,
                },
                'encoders': {
                    'user_encoder': user_encoder,
                    'product_encoder': product_encoder,
                    'user_scaler': user_scaler,
                    'product_scaler': product_scaler
                },
                'mappings': {
                    'user_id_to_idx': user_id_to_idx,
                    'product_id_to_idx': product_id_to_idx
                }
            }
    
    def prepare_lightfm_data(self):
            """
            تهيئة بيانات LightFM من مخرجات load_and_preprocess_data()

            Returns:
                tuple: (interactions_matrix, user_features_matrix, item_features_matrix)
            """
            # 1. إنشاء Dataset
            dataset = dtset(user_identity_features=True, item_identity_features=True)

            # 2. إعداد المستخدمين والمنتجات
            unique_users = self.data['interactions']['user_id'].unique()
            unique_items = self.data['interactions']['product_id'].unique()

            # 3. تجهيز الميزات
            user_features = []
            item_features = []

            # الميزات العددية للمستخدمين (العمر، الدخل)
            for user_id in unique_users:
                user_row = self.data['users'][self.data['users']['user_id'] == user_id].iloc[0]
                features = list(user_row[['age']].values)
                user_features.append((user_id, features))

            encoded_df = pd.DataFrame(
                        self.data['features']['product_categorical'],
                        columns=self.data['encoders']['product_encoder'].get_feature_names_out(['category', 'brand'])
                        )

            products = self.data['products'].drop(columns=['category', 'brand'])
            products = pd.concat([products, encoded_df], axis=1)
            # الميزات الفئوية للمنتجات (الفئة، الماركة، الدولة)
            for product_id in unique_items:
                product_row = products[products['product_id'] == product_id].iloc[0]
                features = list(product_row.filter(regex='^(category_|brand_)').values)
                item_features.append((product_id, features))

            # 4. تهيئة Dataset
            dataset.fit(
                users=unique_users,
                items=unique_items,
                user_features=set(feat for _, feats in user_features for feat in feats),
                item_features=set(feat for _, feats in item_features for feat in feats)
            )

            # 5. بناء المصفوفات
            (interactions_matrix, _) = dataset.build_interactions(
                (row['user_id'], row['product_id'], row['rating'])
                for _, row in self.data['interactions'].iterrows()
            )

            user_features_matrix = dataset.build_user_features(user_features)
            item_features_matrix = dataset.build_item_features(item_features)

            return interactions_matrix, user_features_matrix, item_features_matrix
    
    def train_lightfm(self):
        """
        تدريب نموذج LightFM على البيانات المعالجة
        """
        try:
            # 1. تحضير البيانات
            interactions, user_features, item_features = self.prepare_lightfm_data()

            # 2. إنشاء النموذج
            model = LightFM(
                loss='warp',
                no_components=30,
                learning_rate=0.05,
                item_alpha=0.001,
                user_alpha=0.001
            )

            # 3. التدريب
            model.fit(
                interactions,
                user_features=user_features,
                item_features=item_features,
                epochs=20,
                num_threads=4,
                verbose=True
            )
             # حفظ النموذج
            with open(Config.DATA_PATHS['lightfm_model'], 'wb') as f:
                pickle.dump(model, f)

        except Exception as e:
                print(f"خطأ في تدريب LightFM: {e}")
                raise
    

    def load_model(self):
        try:
           
            # تحميل نموذج LightFM
            if os.path.exists(Config.DATA_PATHS['lightfm_model']):
                with open(Config.DATA_PATHS['lightfm_model'], 'rb') as f:
                    self.model = pickle.load(f)
                
        except Exception as e:
             self.logger.error(f"خطأ في تحميل النماذج: {e}")


    def recommend_with_lightfm(self, user_id, product_id,context=None):
        try:
                            
                            # 1. تحضير بيانات الميزات
                            _, user_features, item_features = self.prepare_lightfm_data()

                            # 2. الحصول على مؤشرات المستخدم والمنتج
                            user_map = {user: idx for idx, user in enumerate(self.data['interactions']['user_id'].unique())}
                            item_map = {item: idx for idx, item in enumerate(self.data['interactions']['product_id'].unique())}
                            user_idx = user_map[user_id]
                            item_idx = item_map[product_id]
                            if isinstance(item_idx, int):
                                    item_idx = [item_idx]
                            # 3. التنبؤ بالدرجة
                            score = self.model.predict(
                                    user_ids=user_idx,
                                    item_ids=np.int32(item_idx),
                                    user_features=user_features,
                                    item_features=item_features,
                                    num_threads=8
                                        )
                            # تحويل من [-1, 1] إلى [0, 1]
                            lightfm_pred = float((score[0] + 1) / 2)
                            return lightfm_pred
        except KeyError:
                                # print(f"تحذير: المستخدم {user_id} أو المنتج {product_id} غير موجود في بيانات التدريب")
                                lightfm_pred=0.3  # قيمة افتراضية إذا لم يوجد التفاعل
                                return lightfm_pred

        except Exception as e:
                                print(f"خطأ في التنبؤ: {e}")
                                raise         