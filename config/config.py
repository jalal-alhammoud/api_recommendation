import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    
    # إعدادات النماذج
    MODEL_PARAMS = {
        'svd': {
            'n_factors': 20,
            'n_epochs': 15,
            'lr_all': 0.005,
            'reg_all': 0.02
        },
        'lightfm': {
            'no_components': 30,
            'loss': 'warp',
            'learning_rate': 0.05,
            'item_alpha': 0.001
        },
        'hybrid': {
            'user_dim': 15,  # بعد تضمين المستخدم
            'product_dim': 50,  # بعد تضمين المنتج
            'hidden_units': 256
        }
    }
    
    # مسارات الملفات
    DATA_PATHS = {
                # ملفات البيانات الأساسية
            'users': os.path.join(BASE_DIR, 'data', 'users.csv'),
            'products': os.path.join(BASE_DIR, 'data', 'products.csv'),
            'reviews': os.path.join(BASE_DIR, 'data', 'reviews.csv'),
            'interactions': os.path.join(BASE_DIR, 'data', 'interactions.csv'),
            'context': os.path.join(BASE_DIR, 'data', 'context.csv'),
            'training_data_merged': os.path.join(BASE_DIR, 'data', 'training_data_merged.csv'),
            
            # مسارات متعلقة بالصور
            'images_df': os.path.join(BASE_DIR, 'data', 'image_data', 'images_data.csv'),
            'image_folder': os.path.join(BASE_DIR, 'data', 'image_data', 'images'),
            
            # نماذج التعلم الآلي
            'resnet50': os.path.join(BASE_DIR, 'models', 'resnet50.h5'),
            'hybrid_model': os.path.join(BASE_DIR, 'models', 'hybrid_model.keras'),
            'knn_model': os.path.join(BASE_DIR, 'models', 'knn_model.pkl'),
            'lightfm_model': os.path.join(BASE_DIR, 'models', 'lightfm_model.pkl'),
            'svd_model': os.path.join(BASE_DIR, 'models', 'svd_model.pkl'),
            
            # ملفات البيانات المعالجة
            'image_features': os.path.join(BASE_DIR, 'models', 'data', 'image_features.pkl'),
            'products_processed': os.path.join(BASE_DIR, 'models', 'data', 'products_processed.pkl'),
            
            # ملفات نظام التوصية
            'most_popular': os.path.join(BASE_DIR, 'recommender', 'recommender_data', 'most_popular.csv'),
            'users_with_interests': os.path.join(BASE_DIR, 'recommender', 'recommender_data', 'users_with_interests.csv'),
            'features_clusters': os.path.join(BASE_DIR, 'recommender', 'recommender_data', 'features_clusters.csv'),
            'interactions_clusters': os.path.join(BASE_DIR, 'recommender', 'recommender_data', 'interactions_clusters.csv'),
            'combined_clusters': os.path.join(BASE_DIR, 'recommender', 'recommender_data', 'combined_clusters.csv'),
            'cluster_centers': os.path.join(BASE_DIR, 'recommender', 'recommender_data', 'cluster_centers.csv'),
            'recommendations': os.path.join(BASE_DIR, 'recommender', 'recommender_data', 'recommendations.csv')

    }

    

        # 'users': 'data/users.csv',
        # 'products': 'data/products.csv',
        # 'reviews': 'data/reviews.csv',
        # 'interactions': 'data/interactions.csv',
        # 'context': 'data/context.csv',
        # 'images_df': 'data/image_data/images_data.csv',
        # 'image_folder': 'data/image_data/images',
        # 'resnet50': 'models/resnet50.h5',
        # 'image_features': 'models/data/image_features.pkl',
        # 'products_processed': 'models/data/products_processed.pkl',
        # 'hybrid_model': "models/hybrid_model.keras",
        # 'training_data_merged': 'data/training_data_merged.csv',
        # 'knn_model': 'models/knn_model.pkl',
        # 'lightfm_model': 'models/lightfm_model.pkl',
        # 'svd_model': 'models/svd_model.pkl',
        # 'most_popular': 'recommender/recommender_data/most_popular.csv',
        # 'users_with_interests': 'recommender/recommender_data/users_with_interests.csv',
        # 'features_clusters': 'recommender/recommender_data/features_clusters.csv',
        # 'interactions_clusters': 'recommender/recommender_data/interactions_clusters.csv',
        # 'combined_clusters': 'recommender/recommender_data/combined_clusters.csv',
        # 'cluster_centers': 'recommender/recommender_data/cluster_centers.csv',
        # 'recommendations': 'recommender/recommender_data/recommendations.csv'