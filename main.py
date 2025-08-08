from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
from config.config import Config
from functions.recommendingFuctions import load_and_preprocess_data
# تهيئة تطبيق FastAPI
app = FastAPI(
    title="Recommendation System API",
    description="A comprehensive API for product recommendations using various algorithms",
    version="1.0.0",
    openapi_tags=[{
        "name": "Recommendations",
        "description": "Endpoints for different recommendation algorithms"
    }]
)

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

# نماذج البيانات (Pydantic models)
class UserProductRequest(BaseModel):
    user_id: int
    product_id: int

class UserRequest(BaseModel):
    user_id: int

class ProductRequest(BaseModel):
    product_id: int

class ContentBasedRequest(BaseModel):
    column_name: str
    value: str
    features: Optional[List[str]] = None

class TextBasedRequest(BaseModel):
    text: str
    column_name: str = "review_text"
    df : str = "reviews_df"
    top_n: int = 5

class DeepLearningRequest(BaseModel):
    user_id: int
    product_id: int
    age: int
    gender: str
    price: float
    location_user: str
    review_text: str
    category: str
    brand: str
    time_of_day: str
    device: str
    location_con: str

class ImageRequest(BaseModel):
    image_name: str
    top_n: int = 3

class ClusterRequest(BaseModel):
    features: List[str]
    df : str = 'users'
    n_clusters: int = 10
    auto_optimize_k: bool = False
    max_k: int = 15

class ExplanationRequest(BaseModel):
    user_data: Dict[str, Any]
    product_data: Dict[str, Any]
    recommendation_score: float


class RecommendationResponse(BaseModel):
    product_id: str
    # name: str
    # score: float
    # explanation: str
    # price: float
    # image_url: str

# === نقاط النهاية للوظائف الأساسية ===

@app.post("/svd/recommend", tags=["Recommendations"])
async def recommend_with_svd(request: UserProductRequest):
    """
    Get recommendation using SVD model
    """
    from modelServices.svdModel import SvdModel
    svd = SvdModel()
    try:
        recommendation = svd.recommend_with_svd(request.user_id, request.product_id)
        return {"user_id": request.user_id, "product_id": request.product_id, "prediction": recommendation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knn/recommend", tags=["Recommendations"])
async def recommend_with_knn(request: UserProductRequest):
    """
    Get recommendation using KNN model
    """
    from modelServices.knnModel import KnnModel
    knn = KnnModel()
    try:
        recommendation = knn.recommend_with_knn(request.user_id, request.product_id)
        return {"user_id": request.user_id, "product_id": request.product_id, "prediction": recommendation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lightfm/recommend", tags=["Recommendations"])
async def recommend_with_lightfm(request: UserProductRequest):
    """
    Get recommendation using LightFM model
    """
    from modelServices.lightfmModel import LightfmModel
    lightmodel = LightfmModel()
    try:
        recommendation = lightmodel.recommend_with_lightfm(request.user_id, request.product_id)
        return {"user_id": request.user_id, "product_id": request.product_id, "prediction": recommendation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deeplearning/recommend", tags=["Recommendations"])
async def recommend_with_deeplearning(request: DeepLearningRequest):
    """
    Get recommendation using Deep Learning model
    """
    from modelServices.deepLearnModel import DeepLearnModel
    deeplearn = DeepLearnModel()
    try:
        input_data = request.dict()
        prediction = deeplearn.predict_rating(input_data)
        return {"prediction": prediction, "input_data": input_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

 
@app.post("/image/recommend", tags=["Recommendations"], response_model=List[RecommendationResponse])
async def recommend_by_image(request: ImageRequest):
    """
    Get recommendations based on product image similarity
    """
    from imageRecommender.imageRecommender import ImageRecommendationService
    try:
        recommendations = []
        service = ImageRecommendationService()
        image_recommendations = service.get_recommendations(request.image_name, request.top_n)
        for pro in image_recommendations:
            recommendations.append({
                'product_id': pro['product_id'],
            })
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/content-based/simple", tags=["Recommendations"])
async def content_based_recommendation(request: ContentBasedRequest):
    """
    Get content-based recommendations based on column and value
    """
    from functions.recommendingFuctions import content_based_simple
    try:
        recommendation = []
        recommendation_based = content_based_simple(
            request.column_name,
            request.value,
            products_df,
            request.features if request.features else ['category', 'brand']
        )
        recommendation_titles = recommendation_based.tolist()
        for recom in recommendation_titles:
            recommendation.append({
                request.column_name: recom
            })
        return recommendation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/most-popular/{top_n}", tags=["Recommendations"], response_model=List[RecommendationResponse])
async def most_popular(top_n: int = 10):
    """
    Get most popular products based on user ratings
    """
    from functions.recommendingFuctions import mostPopular
    try:
        most = []
        popular_products = mostPopular(reviews_df, top_n)
        product_ids_list =popular_products['product_id'].tolist()
        for pro_id in product_ids_list:
            most.append({
                'product_id': pro_id,
            })
        return most
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feature-based/knn", tags=["Recommendations"], response_model=List[RecommendationResponse])
async def feature_based_knn_recommendation(column_name: str, value: str, entity_type: str = "product"):
    """
    Get recommendations based on features using KNN
    """
    from functions.recommendingFuctions import get_recomend_with_features_knn
    try:
        if entity_type == "product":
            df = metadata['products']
            pro_num = pd.DataFrame(metadata['features']['product_numerical'])
            pro_cat = pd.DataFrame(metadata['features']['product_categorical'])
        else:  # user
            df = metadata['users']
            pro_num = pd.DataFrame(metadata['features']['user_numerical'])
            pro_cat = pd.DataFrame(metadata['features']['user_categorical'])
        recommendation = []
        features = pd.concat([pro_num, pro_cat], axis=1)
        recommendations_knn = get_recomend_with_features_knn(df, features, column_name, value)
        for p in recommendations_knn:
            recommendation.append({
                'product_id': p.iat[0]
            })
        return recommendation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/content-based/cosine", tags=["Recommendations"], response_model=List[RecommendationResponse])
async def content_based_cosine_recommendation(column_name: str, value: str):
    """
    Get content-based recommendations using cosine similarity
    """
    from functions.recommendingFuctions import get_recomend_with_cosine
    try:
        df = metadata['products']
        pro_num = pd.DataFrame(metadata['features']['product_numerical'])
        pro_cat = pd.DataFrame(metadata['features']['product_categorical'])
        features = pd.concat([pro_num, pro_cat], axis=1)
        recommendations_cosine = get_recomend_with_cosine(df, features, column_name, value)
        recommendations = []
        recommendations_cosine_id = recommendations_cosine['product_id'].tolist()
        for pro_id in recommendations_cosine_id:
            recommendations.append({
                'product_id': pro_id
            })
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text-based/recommend", tags=["Recommendations"],  response_model=List[RecommendationResponse])
async def text_based_recommendation(request: TextBasedRequest):
    """
    Get recommendations based on text description or review
    """
    from functions.recommendingFuctions import get_recomend_with_describition
    try:
        if request.df == "reviews_df":
            recommendations_desc = get_recomend_with_describition(
                reviews_df, request.column_name, request.text, request.column_name
            )
        recommendations_list = recommendations_desc.tolist() 
        recommendations =[]   
        for pro_id in recommendations_list:
            recommendations.append({
                'product_id': pro_id 
            })
        
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cluster-data", tags=["Recommendations"])
async def cluster_data_endpoint(request: ClusterRequest):
    """
    Cluster data based on numerical features
    """
    from functions.recommendingFuctions import cluster_data
    try:
        if request.df == 'users':
            clustered_data = cluster_data(
                users_df,
                features=request.features,
                n_clusters=request.n_clusters,
                auto_optimize_k=request.auto_optimize_k,
                max_k=request.max_k
            )
             # Convert to JSON serializable format
            result = clustered_data[['user_id'] + request.features + ['cluster']].to_dict(orient='records')
        elif request.df == 'products':
            clustered_data = cluster_data(
                products_df,
                features=request.features,
                n_clusters=request.n_clusters,
                auto_optimize_k=request.auto_optimize_k,
                max_k=request.max_k
            )
             # Convert to JSON serializable format
            result = clustered_data[['priduct_id'] + request.features + ['cluster']].to_dict(orient='records')
        else: 
            raise HTTPException(status_code=404, detail="Only users and products can clustered")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/description-based/recommend", tags=["Recommendations"], response_model=List[RecommendationResponse])
async def description_based_recommendation(request: TextBasedRequest):
    """
    Get recommendations based on user description and optional category
    """
    from functions.recommendingFuctions import recommend_products_by_description
    try:
        recommendations_desc = recommend_products_by_description(
            reviews_df,
            user_description=request.text,
            description_column=request.column_name,
            top_n=request.top_n
        )
        recommendation= []
        re_pro = recommendations_desc['product_id'].tolist()
        for pro_id in re_pro:
            recommendation.append({
                'product_id': pro_id
            })
        return recommendation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-interests/{user_id}", tags=["User Analysis"])
async def get_user_interests(user_id: int):
    """
    Extract user interests from their data
    """
    from functions.recommendingFuctions import extract_user_interests
    try:
        users_with_interests = extract_user_interests(users_df, reviews_df, interactions_df, products_df)
        user_data = users_with_interests[users_with_interests['user_id'] == user_id].to_dict(orient='records')
        
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/similar-users_features/{user_id}", tags=["User Analysis"])
async def get_similar_users(user_id: int, n_recommendations: int = 5):
    """
    Find users with similar interests to the target user
    """
    from functions.recommendingFuctions import get_similar_user_from_userfeatures
    try:
        
        similar_users = get_similar_user_from_userfeatures(user_id=user_id)
        return similar_users
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similar-users/{user_id}", tags=["User Analysis"])
async def get_similar_users(user_id: int, n_recommendations: int = 5):
    """
    Find users with similar interests to the target user
    """
    from functions.recommendingFuctions import extract_user_interests, find_similar_users
    try:
        users_with_interests = extract_user_interests(users_df, reviews_df, interactions_df, products_df)
        similar_users = find_similar_users(
            target_user_id=user_id,
            users_with_interests_df=users_with_interests,
            reviews_df=reviews_df,
            interactions_df=interactions_df,
            n_recommendations=n_recommendations
        )
        similar_users_dict = similar_users.to_dict(orient='records')
        return similar_users_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-high-rated/{user_id}", tags=["User Analysis"])
async def get_user_high_rated_products(user_id: int, min_rating: int = 3):
    """
    Get products rated highly by a user
    """
    from functions.recommendingFuctions import get_high_rated_products
    try:
        high_rated = get_high_rated_products(reviews_df, user_id, min_rating)
        return {
            "user_id": user_id,
            "min_rating": min_rating,
            "high_rated_products": high_rated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-interacted-products/{user_id}", tags=["User Analysis"])
async def get_user_interacted_products_endpoint(user_id: int):
    """
    Get products a user has interacted with
    """
    from functions.recommendingFuctions import get_user_interacted_products
    try:
        interacted_products = get_user_interacted_products(interactions_df, user_id)
        return {
            "user_id": user_id,
            "interacted_products": interacted_products
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-preferred-products/{user_id}", tags=["User Analysis"])
async def get_user_preferred_products(user_id: int):
    """
    Get products a user has interacted with or rated highly
    """
    from functions.recommendingFuctions import get_user_prefered_product
    try:
        preferred_products = get_user_prefered_product(reviews_df, interactions_df, user_id)
        return {
            "user_id": user_id,
            "preferred_products": preferred_products
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-interactions-details/{user_id}", tags=["User Analysis"])
async def get_user_interactions_details_endpoint(user_id: int):
    """
    Get all interaction details for a user
    """
    from functions.recommendingFuctions import get_user_interactions_details
    try:
        interactions = get_user_interactions_details(interactions_df, user_id)
        return {
            "user_id": user_id,
            "interactions": interactions.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-interactions-count/{user_id}", tags=["User Analysis"])
async def get_user_interactions_count_endpoint(user_id: int):
    """
    Get interaction counts per product for a user
    """
    from functions.recommendingFuctions import get_user_interactions_count
    try:
        interaction_counts = get_user_interactions_count(interactions_df, user_id)
        return {
            "user_id": user_id,
            "interaction_counts": interaction_counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-interactions-by-type/{user_id}", tags=["User Analysis"])
async def get_user_interactions_by_type_endpoint(user_id: int, interaction_type: str):
    """
    Get user interactions filtered by type
    """
    from functions.recommendingFuctions import get_user_interactions_by_type
    try:
        interactions = get_user_interactions_by_type(interactions_df, user_id, interaction_type)
        return {
            "user_id": user_id,
            "interaction_type": interaction_type,
            "interactions": interactions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/similar-users-products/{user_id}", tags=["Recommendations"])
async def get_similar_users_products(user_id: int):
    """
    Get products preferred by users with similar interests
    """
    from functions.recommendingFuctions import get_user_similar_prefered_products
    try:
        products = get_user_similar_prefered_products(
            user_id, users_df, products_df, interactions_df, reviews_df
        )
        return {
            "user_id": user_id,
            "products_from_similar_users": products
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/product-based-recommendations", tags=["Recommendations"])
async def get_product_based_recommendations(product_ids: List[int]):
    """
    Get recommendations based on specific products
    """
    from functions.recommendingFuctions import get_products_from_product
    try:
        recommendations = get_products_from_product(product_ids)
        return {
            "product_ids": product_ids,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-suggestions/{user_id}", tags=["Recommendations"])
async def get_user_suggestions(user_id: int):
    """
    Get comprehensive suggestions for a user
    """
    from functions.recommendingFuctions import suggest_for_user
    try:
        suggestions = suggest_for_user(user_id)
        return {
            "user_id": user_id,
            "suggestions": suggestions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/deep-user-suggestions/{user_id}", tags=["Recommendations"])
async def get_deep_user_suggestions(user_id: int):
    """
    Get deep analysis suggestions for a user with predicted ratings
    """
    from functions.recommendingFuctions import deep_suggest_for_user
    try:
        suggestions = deep_suggest_for_user(user_id)
        return {
            "user_id": user_id,
            "suggestions": suggestions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-explanation", tags=["Explanation"])
async def generate_explanation(request: ExplanationRequest):
    """
    Generate natural language explanation for a recommendation
    """
    from explainer.explainer import RecommendationExplainer
    try:
        explainer = RecommendationExplainer()
        explanation = explainer.generate_explanation(
            user_data=request.user_data,
            product_data=request.product_data,
            recommendation_score=request.recommendation_score
        )
        
        return {
            "explanation": explanation,
            "user_data": request.user_data,
            "product_data": request.product_data,
            "score": request.recommendation_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# نقطة النهاية للصحة
@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)