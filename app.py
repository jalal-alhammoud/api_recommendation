from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException, NotFound
import pandas as pd
from config.config import Config
from functions.recommendingFuctions import load_and_preprocess_data
from typing import Optional, List, Dict, Any

app = Flask(__name__)

# إعداد معلومات وصفية للAPI
API_METADATA = {
    "title": "Recommendation System API",
    "description": "A comprehensive API for product recommendations using various algorithms",
    "version": "1.0.0"
}

# تحميل البيانات الأساسية
try:
    products_df = pd.read_csv(Config.DATA_PATHS['products'], low_memory=False)
    reviews_df = pd.read_csv(Config.DATA_PATHS['reviews'], low_memory=False)
    interactions_df = pd.read_csv(Config.DATA_PATHS['interactions'], low_memory=False)
    users_df = pd.read_csv(Config.DATA_PATHS['users'], low_memory=False)
    images_df = pd.read_csv(Config.DATA_PATHS['images_df'])
    metadata = load_and_preprocess_data()
except Exception as e:
    print(f"Error loading data: {str(e)}")

# معالجة الأخطاء
@app.errorhandler(404)
def not_found(e):
    return jsonify(error=str(e)), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify(error=str(e)), 500

# === نقاط النهاية للوظائف الأساسية ===

@app.route("/svd/recommend", methods=["POST"])
def recommend_with_svd():
    """Get recommendation using SVD model"""
    from modelServices.svdModel import SvdModel
    svd = SvdModel()
    try:
        data = request.get_json()
        recommendation = svd.recommend_with_svd(data['user_id'], data['product_id'])
        return jsonify({
            "user_id": data['user_id'],
            "product_id": data['product_id'],
            "prediction": recommendation
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/knn/recommend", methods=["POST"])
def recommend_with_knn():
    """Get recommendation using KNN model"""
    from modelServices.knnModel import KnnModel
    knn = KnnModel()
    try:
        data = request.get_json()
        recommendation = knn.recommend_with_knn(data['user_id'], data['product_id'])
        return jsonify({
            "user_id": data['user_id'],
            "product_id": data['product_id'],
            "prediction": recommendation
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/lightfm/recommend", methods=["POST"])
def recommend_with_lightfm():
    """Get recommendation using LightFM model"""
    from modelServices.lightfmModel import LightfmModel
    lightmodel = LightfmModel()
    try:
        data = request.get_json()
        recommendation = lightmodel.recommend_with_lightfm(data['user_id'], data['product_id'])
        return jsonify({
            "user_id": data['user_id'],
            "product_id": data['product_id'],
            "prediction": recommendation
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/deeplearning/recommend", methods=["POST"])
def recommend_with_deeplearning():
    """Get recommendation using Deep Learning model"""
    from modelServices.deepLearnModel import DeepLearnModel
    deeplearn = DeepLearnModel()
    try:
        input_data = request.get_json()
        prediction = deeplearn.predict_rating(input_data)
        return jsonify({
            "prediction": prediction,
            "input_data": input_data
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/image/recommend", methods=["POST"])
def recommend_by_image():
    """Get recommendations based on product image similarity"""
    from imageRecommender.imageRecommender import ImageRecommendationService
    try:
        data = request.get_json()
        recommendations = []
        service = ImageRecommendationService()
        image_recommendations = service.get_recommendations(data['image_name'], data.get('top_n', 3))
        for pro in image_recommendations:
            recommendations.append({
                'product_id': pro['product_id'],
            })
        return jsonify(recommendations)
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/content-based/simple", methods=["POST"])
def content_based_recommendation():
    """Get content-based recommendations based on column and value"""
    from functions.recommendingFuctions import content_based_simple
    try:
        data = request.get_json()
        recommendation = []
        recommendation_based = content_based_simple(
            data['column_name'],
            data['value'],
            products_df,
            data.get('features', ['category', 'brand'])
        )
        recommendation_titles = recommendation_based.tolist()
        for recom in recommendation_titles:
            recommendation.append({
                data['column_name']: recom
            })
        return jsonify(recommendation)
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/most-popular/<int:top_n>", methods=["GET"])
def most_popular(top_n):
    """Get most popular products based on user ratings"""
    from functions.recommendingFuctions import mostPopular
    try:
        most = []
        popular_products = mostPopular(reviews_df, top_n)
        product_ids_list = popular_products['product_id'].tolist()
        for pro_id in product_ids_list:
            most.append({
                'product_id': pro_id,
            })
        return jsonify(most)
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/feature-based/knn", methods=["POST"])
def feature_based_knn_recommendation():
    """Get recommendations based on features using KNN"""
    from functions.recommendingFuctions import get_recomend_with_features_knn
    try:
        data = request.get_json()
        if data.get('entity_type', 'product') == "product":
            df = metadata['products']
            pro_num = pd.DataFrame(metadata['features']['product_numerical'])
            pro_cat = pd.DataFrame(metadata['features']['product_categorical'])
        else:  # user
            df = metadata['users']
            pro_num = pd.DataFrame(metadata['features']['user_numerical'])
            pro_cat = pd.DataFrame(metadata['features']['user_categorical'])
        recommendation = []
        features = pd.concat([pro_num, pro_cat], axis=1)
        recommendations_knn = get_recomend_with_features_knn(df, features, data['column_name'], data['value'])
        for p in recommendations_knn:
            recommendation.append({
                'product_id': p.iat[0]
            })
        return jsonify(recommendation)
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/content-based/cosine", methods=["POST"])
def content_based_cosine_recommendation():
    """Get content-based recommendations using cosine similarity"""
    from functions.recommendingFuctions import get_recomend_with_cosine
    try:
        data = request.get_json()
        df = metadata['products']
        pro_num = pd.DataFrame(metadata['features']['product_numerical'])
        pro_cat = pd.DataFrame(metadata['features']['product_categorical'])
        features = pd.concat([pro_num, pro_cat], axis=1)
        recommendations_cosine = get_recomend_with_cosine(df, features, data['column_name'], data['value'])
        recommendations = [{'product_id': pid} for pid in recommendations_cosine['product_id'].tolist()]
        return jsonify(recommendations)
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/text-based/recommend", methods=["POST"])
def text_based_recommendation():
    """Get recommendations based on text description or review"""
    from functions.recommendingFuctions import get_recomend_with_describition
    try:
        data = request.get_json()
        if data.get('df', 'reviews_df') == "reviews_df":
            recommendations_desc = get_recomend_with_describition(
                reviews_df, data['column_name'], data['text'], data['column_name']
            )
        recommendations = [{'product_id': pid} for pid in recommendations_desc.tolist()]
        return jsonify(recommendations)
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/cluster-data", methods=["POST"])
def cluster_data_endpoint():
    """Cluster data based on numerical features"""
    from functions.recommendingFuctions import cluster_data
    try:
        data = request.get_json()
        if data['df'] == 'users':
            clustered_data = cluster_data(
                users_df,
                features=data['features'],
                n_clusters=data.get('n_clusters', 10),
                auto_optimize_k=data.get('auto_optimize_k', False),
                max_k=data.get('max_k', 15)
            )
            result = clustered_data[['user_id'] + data['features'] + ['cluster']].to_dict(orient='records')
        elif data['df'] == 'products':
            clustered_data = cluster_data(
                products_df,
                features=data['features'],
                n_clusters=data.get('n_clusters', 10),
                auto_optimize_k=data.get('auto_optimize_k', False),
                max_k=data.get('max_k', 15)
            )
            result = clustered_data[['product_id'] + data['features'] + ['cluster']].to_dict(orient='records')
        else:
            raise NotFound("Only users and products can be clustered")
        return jsonify(result)
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/description-based/recommend", methods=["POST"])
def description_based_recommendation():
    """Get recommendations based on user description"""
    from functions.recommendingFuctions import recommend_products_by_description
    try:
        data = request.get_json()
        recommendations_desc = recommend_products_by_description(
            reviews_df,
            user_description=data['text'],
            description_column=data.get('column_name', 'review_text'),
            top_n=data.get('top_n', 5)
        )
        recommendations = [{'product_id': pid} for pid in recommendations_desc['product_id'].tolist()]
        return jsonify(recommendations)
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/user-interests/<int:user_id>", methods=["GET"])
def get_user_interests(user_id):
    """Extract user interests from their data"""
    from functions.recommendingFuctions import extract_user_interests
    try:
        users_with_interests = extract_user_interests(users_df, reviews_df, interactions_df, products_df)
        user_data = users_with_interests[users_with_interests['user_id'] == user_id].to_dict(orient='records')
        if not user_data:
            raise NotFound("User not found")
        return jsonify(user_data)
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/similar-users_features/<int:user_id>", methods=["GET"])
def get_similar_users_features(user_id):
    """Find users with similar interests to the target user"""
    from functions.recommendingFuctions import get_similar_user_from_userfeatures
    try:
        similar_users = get_similar_user_from_userfeatures(user_id=user_id)
        return jsonify(similar_users)
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/similar-users/<int:user_id>", methods=["GET"])
def get_similar_users(user_id):
    """Find users with similar interests to the target user"""
    from functions.recommendingFuctions import extract_user_interests, find_similar_users
    try:
        n_recommendations = request.args.get('n_recommendations', default=5, type=int)
        users_with_interests = extract_user_interests(users_df, reviews_df, interactions_df, products_df)
        similar_users = find_similar_users(
            target_user_id=user_id,
            users_with_interests_df=users_with_interests,
            reviews_df=reviews_df,
            interactions_df=interactions_df,
            n_recommendations=n_recommendations
        )
        return jsonify(similar_users.to_dict(orient='records'))
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": API_METADATA["version"],
        "api_title": API_METADATA["title"],
        "description": API_METADATA["description"]
    })

@app.route('/user-high-rated/<int:user_id>', methods=['GET'])
def get_user_high_rated_products(user_id):
    """Get products rated highly by a user"""
    from functions.recommendingFuctions import get_high_rated_products
    try:
        min_rating = request.args.get('min_rating', default=3, type=int)
        high_rated = get_high_rated_products(reviews_df, user_id, min_rating)
        return jsonify({
            "user_id": user_id,
            "min_rating": min_rating,
            "high_rated_products": high_rated
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/user-interacted-products/<int:user_id>', methods=['GET'])
def get_user_interacted_products_endpoint(user_id):
    """Get products a user has interacted with"""
    from functions.recommendingFuctions import get_user_interacted_products
    try:
        interacted_products = get_user_interacted_products(interactions_df, user_id)
        return jsonify({
            "user_id": user_id,
            "interacted_products": interacted_products
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/user-preferred-products/<int:user_id>', methods=['GET'])
def get_user_preferred_products(user_id):
    """Get products a user has interacted with or rated highly"""
    from functions.recommendingFuctions import get_user_prefered_product
    try:
        preferred_products = get_user_prefered_product(reviews_df, interactions_df, user_id)
        return jsonify({
            "user_id": user_id,
            "preferred_products": preferred_products
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/user-interactions-details/<int:user_id>', methods=['GET'])
def get_user_interactions_details_endpoint(user_id):
    """Get all interaction details for a user"""
    from functions.recommendingFuctions import get_user_interactions_details
    try:
        interactions = get_user_interactions_details(interactions_df, user_id)
        return jsonify({
            "user_id": user_id,
            "interactions": interactions.to_dict(orient='records')
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/user-interactions-count/<int:user_id>', methods=['GET'])
def get_user_interactions_count_endpoint(user_id):
    """Get interaction counts per product for a user"""
    from functions.recommendingFuctions import get_user_interactions_count
    try:
        interaction_counts = get_user_interactions_count(interactions_df, user_id)
        return jsonify({
            "user_id": user_id,
            "interaction_counts": interaction_counts
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/user-interactions-by-type/<int:user_id>', methods=['GET'])
def get_user_interactions_by_type_endpoint(user_id):
    """Get user interactions filtered by type"""
    from functions.recommendingFuctions import get_user_interactions_by_type
    try:
        interaction_type = request.args.get('interaction_type', type=str)
        interactions = get_user_interactions_by_type(interactions_df, user_id, interaction_type)
        return jsonify({
            "user_id": user_id,
            "interaction_type": interaction_type,
            "interactions": interactions
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/similar-users-products/<int:user_id>', methods=['GET'])
def get_similar_users_products(user_id):
    """Get products preferred by users with similar interests"""
    from functions.recommendingFuctions import get_user_similar_prefered_products
    try:
        products = get_user_similar_prefered_products(
            user_id, users_df, products_df, interactions_df, reviews_df
        )
        return jsonify({
            "user_id": user_id,
            "products_from_similar_users": products
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/product-based-recommendations', methods=['POST'])
def get_product_based_recommendations():
    """Get recommendations based on specific products"""
    from functions.recommendingFuctions import get_products_from_product
    try:
        data = request.get_json()
        product_ids = data.get('product_ids', [])
        recommendations = get_products_from_product(product_ids)
        return jsonify({
            "product_ids": product_ids,
            "recommendations": recommendations
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/user-suggestions/<int:user_id>', methods=['GET'])
def get_user_suggestions(user_id):
    """Get comprehensive suggestions for a user"""
    from functions.recommendingFuctions import suggest_for_user
    try:
        suggestions = suggest_for_user(user_id)
        return jsonify({
            "user_id": user_id,
            "suggestions": suggestions
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/deep-user-suggestions/<int:user_id>', methods=['GET'])
def get_deep_user_suggestions(user_id):
    """Get deep analysis suggestions for a user with predicted ratings"""
    from functions.recommendingFuctions import deep_suggest_for_user
    try:
        suggestions = deep_suggest_for_user(user_id)
        return jsonify({
            "user_id": user_id,
            "suggestions": suggestions
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/generate-explanation', methods=['POST'])
def generate_explanation():
    """Generate natural language explanation for a recommendation"""
    from explainer.explainer import RecommendationExplainer
    try:
        data = request.get_json()
        explainer = RecommendationExplainer()
        explanation = explainer.generate_explanation(
            user_data=data['user_data'],
            product_data=data['product_data'],
            recommendation_score=data['recommendation_score']
        )
        return jsonify({
            "explanation": explanation,
            "user_data": data['user_data'],
            "product_data": data['product_data'],
            "score": data['recommendation_score']
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)