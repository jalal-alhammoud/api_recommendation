from config.config import Config


###############################################################################
# التنبؤ باستخدام نموذج svd
from modelServices.svdModel import SvdModel
svd= SvdModel()
# دالة التنبؤ recommend_with_svd(user_id, product_id)
# نتيجة التنبؤ هي بين 0 و 5
print("Svd model recommendation:")
svd_recommend = svd.recommend_with_svd(1,5)
print(svd_recommend)
print("############################")

# لإعادة تدريب النموذج
# svd.train_svd()


#####################################################################################3
#التنبؤ باستخدام خوارزمية knn
#يمثل هذا النموذج مثال على التصفية التعاونية القائمة على المستخدم

from modelServices.knnModel import KnnModel
knn= KnnModel()
# دالة التنبؤ recommend_with_knn(user_id, product_id)
# نتيجة التنبؤ هي بين 0 و 5
print("Knn model recommendation:")
knn_recommend = knn.recommend_with_knn(1,5)
print(knn_recommend)
print("############################")


# لإعادة تدريب النموذج
#لتعديل الخيارات
#تغيير مقياس التشابه إلى 'pearson' أو 'msd'
# تجربة user_based=False للتصفية المعتمدة على العناصر
# تعديل عدد الجيران
# train_knn(self,similarity='cosine' ,user_based=True, k=30)
# knn.train_knn()


##################################################################################
# التنبؤ باستخدام lightfm model

# from modelServices.lightfmModel import LightfmModel

# lightmodel = LightfmModel()
# # دالة التنبؤ recommend_with_lightfm(user_id, product_id)
# # نتيجة التنبؤ هي رقم بين 0 و 1
# print("predication With LightFm:")
# predicationWithLightFm = lightmodel.recommend_with_lightfm(1,5)
# print(predicationWithLightFm)
# print("########################")
# # لإعادة تدريب النموذج
# # lightmodel.train_lightfm()

###################################################################################

# التبؤ باستخدام النموذج الهجين (نموذج التعلم العميق)
from modelServices.deepLearnModel import DeepLearnModel

deeplearn = DeepLearnModel()


#لتدريب النموذج وحفظه
# deeplearn.train_and_save_model()

# التبؤ بقيمة دخل مثل 
# خرج التبؤ احتمالي بين 0 و 1
input_data = {
    'user_id': 500,
    'product_id': 10,
    'age': 35,
    'gender': "Female",
    'price': 899.99,
    'location_user': 'Chicago',
    'review_text': "i love this pro",
    'category': "Clothing",
    'brand': "BrandB",
    'time_of_day': "Night",
    'device': "Mobile",
    'location_con': "New York"
}
print('predication with deep learn model:')
pred= deeplearn.predict_rating(input_data)
print(pred)
print("###############################")
#لتحميل النموذج واستمرار التدريب على بيانات جديدة
# deeplearn.load_and_continue_training()

##################################################################3

# التوصية باستخدام الصور الخاصة بالمنتجات بناء على صور المنتجات التي نالت إعجاب أو تفاعل
from imageRecommender.imageRecommender import ImageRecommendationService, prepare_resources
import pandas as pd
df = pd.read_csv(Config.DATA_PATHS['images_df']) # هيكل بيانات يحوي معرفات المنتجات وأسماء الصور

   
# حفظ النموذج والمتجهات مسبقاً (مرة واحدة)
# خطوة تحضيرية (تجرى مرة واحدة)
# prepare_resources(df, image_path='data/image_data/images')

# تهيئة الخدمة
service = ImageRecommendationService()

#التبؤ بالمنتحات
print("recommendations by product image:")
recommendations = service.get_recommendations("fa8e22d6-c0b6-5229-bb9e-ad52eda39a0a.jpg", 3)

print(recommendations) # [{'product_id': 'fa8e22d6-c0b6-5229-bb9e-ad52eda39a0a', 'images': 'fa8e22d6-c0b6-5229-bb9e-ad52eda39a0a.jpg'}, {'product_id': '3f3f97bb-5faf-57df-a9ff-1af24e2b1045', 'images': '3f3f97bb-5faf-57df-a9ff-1af24e2b1045.jpg'}, {'product_id': '2cc3eb4c-42b5-5bc5-9eb6-0f438f45dd58', 'images': '2cc3eb4c-42b5-5bc5-9eb6-0f438f45dd58.jpg'}]
print("###############################")
##########################################################################

#############################


#تابع توصية معتمد على المحتوى يعطي توصيات بناء على اسم عمود والخاصية ا
#يأخذ اسم العمود والخاصية كمدخل ويعطي اقتراحات بناء على الخواص الأخرى 
from functions.recommendingFuctions import content_based_simple,clean

# # Load the dataset 
metadata = pd.read_csv(Config.DATA_PATHS['products'], low_memory=False)
print("نتيجة تابع توصية معتمد على المحتوى يعطي توصيات بناء على اسم العمود والخاصية")
recommendation = content_based_simple('category','Sports', metadata, ['category', 'brand'])
print(recommendation)


########################################

#تابع يقوم باعادة المنتجات الأكثر شعبية
#يعتمد التابع على تقييمات المستخدمين للمنتج

from functions.recommendingFuctions import mostPopular

metadata = pd.read_csv(Config.DATA_PATHS['reviews'], low_memory=False)
predication = mostPopular(metadata,11)
print("إيجاد المنتجات الأكثر شعبية")
print(predication)
print("##################################")

#############################################

#تابع يقوم بالتوصية بمنتج معين أو مستخدم مشابه أو الاثنين معا بناء على المميزات
#يجب ان تكون المميزات على شكل onehotencoding H, scaling للحصول على نتائج
# يتم استخدام مقياس min max لتقليل القيم 
# يتم استخدام KNN للتوصية
# المميزات features على شكل dataframe
# يمكن تعديل التابع بحيث يتم حفظ النموذج وإعادة استخدامه

from functions.recommendingFuctions import get_recomend_with_features_knn,load_and_preprocess_data


metadata = load_and_preprocess_data()
df = metadata['products']
pro_num =pd.DataFrame(metadata['features']['product_numerical'])
pro_cat =pd.DataFrame(metadata['features']['product_categorical'])

features = pd.concat([pro_num,pro_cat], axis=1)
predication = get_recomend_with_features_knn(df, features,'category','Clothing')
print(" نتيجة تابع يقوم بالتوصية بمنتج معين  بناء على المميزات ")
print(predication)
# #مثال على user
df = metadata['users']
pro_num =pd.DataFrame(metadata['features']['user_numerical'])
pro_cat =pd.DataFrame(metadata['features']['user_categorical'])

features = pd.concat([pro_num,pro_cat], axis=1)
predication = get_recomend_with_features_knn(df, features,'location',"Houston")
print(" نتيجة تابع يقوم بالتوصية مستخدم مشابه بناء على المميزات ")
print(predication)
print("##################################")

####################################################

#تابع يقوم بالتوصية بناء على المحتوى باستخدام cosine_similitry
from functions.recommendingFuctions import get_recomend_with_cosine
metadata = load_and_preprocess_data()
df = metadata['products']
pro_num =pd.DataFrame(metadata['features']['product_numerical'])
pro_cat =pd.DataFrame(metadata['features']['product_categorical'])

features = pd.concat([pro_num,pro_cat], axis=1)
predication = get_recomend_with_cosine(df, features,'category','Clothing')
print(" نتيجة تابع يقوم بالتوصية بناء على المحتوى باستخدام cosine_similitry ")
print(predication)
print("##################################")


#########################################################
# تابع يقوم بالتوصية بناء على المحتوى لمنتج معين بناء على وصف أو مراجعة نصية لهذا المنتج

from functions.recommendingFuctions import get_recomend_with_describition
metadata = pd.read_csv(Config.DATA_PATHS['reviews'], low_memory=False)
predication =get_recomend_with_describition(metadata, "product_id",393,'review_text' )
predication =get_recomend_with_describition(metadata, "review_text","Not worth the price.",'review_text' )

print(" نتيجة  تابع يقوم بالتوصية بناء على المحتوى لمنتج معين بناء على وصف أو مراجعة نصية لهذا المنتج ")
print(predication)
print("##################################")

#####################################################
#  دالة لترتيب المنتجات وإرجاع أحدث n منتج
# from functions.recommendingFuctions import get_recomend_with_describition
# get_top_n_products_by_weight(df, date_column, n)



################################################################

# دالة تجميع لهياكل البيانات حسب قيم رقمية لأعمدة معينة مثل العمر السعر 
# cluster_data(
#     df: pd.DataFrame,
#     features: list,
#     n_clusters: int = 10,
#     auto_optimize_k: bool = False,
#     max_k: int = 15,
#     random_state: int = 42
# )
from functions.recommendingFuctions import cluster_data, load_and_preprocess_data

metadata = load_and_preprocess_data()
df = metadata['users']

clustered_df = cluster_data(df, features=['age'], n_clusters=3)
print("# دالة تجميع لهياكل البيانات حسب قيم رقمية لأعمدة معينة مثل العمر السعر ")
print(clustered_df[['user_id', 'age', 'cluster']])

###########################################################
#دلة تقوم باقتراح توصيات بناء على وصف يدخله المستخدم وفئة (اختياري) 
from functions.recommendingFuctions import recommend_products_by_description


df_review = pd.read_csv(Config.DATA_PATHS['reviews'], low_memory=False)

print("#دلة تقوم باقتراح توصيات بناء على وصف يدخله المستخدم وفئة (اختياري) ")
re= recommend_products_by_description(df_review, user_description="I need a phone with great camera quality",description_column="review_text",top_n=5 )

print(re)
print("##################################")

###########################################################

#  تابع لاستنتاج اهتمامات المستخدم من البيانات
    
#     Args:
#         reviews_df (DataFrame): يحتوي على user_id, product_id, rating, review_text
#         products_df (DataFrame): يحتوي على product_id, category, price, brand
#         user_review (str): نص المراجعة من المستخدم
#         user_category (str): الفئة المطلوبة
#         top_n (int): عدد التوصيات المطلوبة
        
#     Returns:
#         DataFrame: المنتجات الموصى بها مع معلوماتها


from functions.recommendingFuctions import extract_user_interests
df_user = pd.read_csv(Config.DATA_PATHS['users'], low_memory=False)
df_pro = pd.read_csv(Config.DATA_PATHS['products'], low_memory=False)
df_inter = pd.read_csv(Config.DATA_PATHS['interactions'], low_memory=False)
df_review = pd.read_csv(Config.DATA_PATHS['reviews'], low_memory=False)

users_with_interests = extract_user_interests(df_user, df_review, df_inter, df_pro)
print("استنتاج اهتمامات المستخدمين من البيانات")
print(users_with_interests.head())
print("##################################")

##################################################################



#  تابع لاستنتاج اهتمامات المستخدم من البيانات
# مع تابع لإيجاد المستخدمين الذين لهم اهتمامات مشتركة 
from functions.recommendingFuctions import extract_user_interests, find_similar_users
df_user = pd.read_csv(Config.DATA_PATHS['users'], low_memory=False)
df_pro = pd.read_csv(Config.DATA_PATHS['products'], low_memory=False)
df_inter = pd.read_csv(Config.DATA_PATHS['interactions'], low_memory=False)
df_review = pd.read_csv(Config.DATA_PATHS['reviews'], low_memory=False)

users_with_interests = extract_user_interests(df_user, df_review, df_inter, df_pro)

similar_users = find_similar_users(
    target_user_id=123,
    users_with_interests_df=users_with_interests,
    reviews_df=df_review,
    interactions_df=df_inter,
    n_recommendations=5
)

# عرض النتائج
print("ايجاد المستخدمين المشابهين بالاهتمامات لمستخدم معين")
print(similar_users[['user_id', 'age', 'gender', 'similarity_score', 
                   'interests_categories', 'preferred_brands']])

print("##################################")

##########################################################################

# تابع يقوم بإرجاع المنتجات التي قام المستخدم بتقيمها والتي هي أعلى من تقييم معين 
from functions.recommendingFuctions import get_high_rated_products

df_review = pd.read_csv(Config.DATA_PATHS['reviews'], low_memory=False)

print("المنتجات التي قام المستخدم  رقم 5 بتقيمها بتفييم أكبر أو يساوي 3")
high_rated_product_for_5 = get_high_rated_products(df_review, 5, r=3)
print(high_rated_product_for_5)
print("################################################")

################################################################################
# تابع يقوم بإرجاع المنتجات التي قام المستخدم بالتفاعل معها

from functions.recommendingFuctions import get_user_interacted_products
df_inter = pd.read_csv(Config.DATA_PATHS['interactions'], low_memory=False)

print("المنتجات التي قام المستخدم 5 بالتفاعل معها إضافة للسلة مشاهدة شراء")
user_inter= get_user_interacted_products(df_inter, 5)
print(user_inter)
print("################################################")

###################################################################################

# تابع يقوم بإرجاع المنتجات التي قام المستخدم بالتفاعل معها أو تقيمها تقييم مرتفع
from functions.recommendingFuctions import get_user_prefered_product
df_review = pd.read_csv(Config.DATA_PATHS['reviews'], low_memory=False)
df_inter = pd.read_csv(Config.DATA_PATHS['interactions'], low_memory=False)

print("المنتجات التي قام المستخدم رقم 5 بالتفاعل معها أو تقييمها")
user_pro = get_user_prefered_product(df_review,df_inter ,5 )
print(user_pro)
print("################################################")

####################################################################################

#إرجاع DataFrame كامل مع جميع التفاعلات:

from functions.recommendingFuctions import  get_user_interactions_details
df_inter = pd.read_csv(Config.DATA_PATHS['interactions'], low_memory=False)
print("كافة تفاعلات المستخدم رقم 5 مع البيانات")

user_inter_df = get_user_interactions_details(df_inter, 5)
print(user_inter_df.head())
print("################################################")

#######################################################################################

# إرجاع عدد التفاعلات لكل منتج:
from functions.recommendingFuctions import  get_user_interactions_count

df_inter = pd.read_csv(Config.DATA_PATHS['interactions'], low_memory=False)
print("عدد تفاعلات المستخدم مع كل منتج")
product_inter= get_user_interactions_count(df_inter, 5)
print(product_inter)

print("################################################")

#############################################################################################

# تصفية حسب نوع التفاعل
from functions.recommendingFuctions import  get_user_interactions_by_type

df_inter = pd.read_csv(Config.DATA_PATHS['interactions'], low_memory=False)
print("تصفية تفاعلات المستخدم حسب نوع التفاعل مشاهدة مثلا")

inter_by_type=  get_user_interactions_by_type(df_inter, 5, 'view')
print(inter_by_type)
print("################################################")

##################################################################################################

#تابع يقوم بإرجاع معرفات المنتجات التي اهتم بها المستخدمين المشابهين بالاهتمامات للمستخدم
# في حال كان المستخدم جديد وغير معروف اهتماماته بعد بسبب عدم التفاعل مع الموقع 
#يتم إرجاع المنتجات للمستخدمين المشابهين بالمواصفات الشخصية لهذا المستخدم
from functions.recommendingFuctions import get_user_similar_prefered_products

df_user = pd.read_csv(Config.DATA_PATHS['users'], low_memory=False)
df_pro = pd.read_csv(Config.DATA_PATHS['products'], low_memory=False)
df_inter = pd.read_csv(Config.DATA_PATHS['interactions'], low_memory=False)
df_review = pd.read_csv(Config.DATA_PATHS['reviews'], low_memory=False)

print("المنتجات التي قام المستخدمون المشابهون بالاهتمامات مع المستخدم 5 بالتفاعل معها")
product_from_similar_users = get_user_similar_prefered_products(5,df_user,df_pro,df_inter,df_review)
print(product_from_similar_users)
print("################################################")

####################################################################################################

# تابع يرجع المنتجات المقترحة بناء على لمنتجات معينة بناء على التوابع السابقة
from functions.recommendingFuctions import get_products_from_product

print("المنتجات المقترحة بناء على التشابه مع المنتج رقم 100 وفق مختلف الخوارزميات السابقة")

recommendation_product = get_products_from_product([100])

print(recommendation_product)

print("################################################")


######################################################################################################
## تابع يعطي اقتراحات لمستخدم معين
from functions.recommendingFuctions import suggest_for_user

print('المنتجات المقترحة للمستخدم رقم 5 اعتمادا على التوابع السابقة')
recommendation_for_user_5 = suggest_for_user(5)

print(recommendation_for_user_5)
print("################################################")

#########################################################################################################
##  تابع يعطي اقتراحات لمستخدم معين مع تحليل عميق

from functions.recommendingFuctions import deep_suggest_for_user
print('المنتجات المقترحة للمستخدم رقم 5 اعتمادا على التوابع السابقة مع تحليل عميق والتنبؤ بتقييم هذا المستخدم لهذه المنتجات')


deep_recommendation = deep_suggest_for_user(5)

print(deep_recommendation)

print("################################################")

#############################################################################################################
# تابع يقوم بتوليد توصيات لكافة المستخدمين
# يستغرق وقت للتنفيذ بشكل كامل
# from functions.recommendingFuctions import generate_recommendations_for_all_users

# recommendation_for_all user  = generate_recommendations_for_all_users()

# print("توليد التوصيات لكل المستخدمين")
# print(recommendation_for_all user.head())

# print("################################################")


################################################################

#تابع يقوم بتجميع المستخدمين ضمن مجموعات حسب مواصفاتهم وحسب اهتماماتهم وحسب المواصفات والاهتمامات معاً
from functions.recommendingFuctions import cluster_all_users
users_df = pd.read_csv(Config.DATA_PATHS['users'])
reviews_df = pd.read_csv(Config.DATA_PATHS['reviews'])
interactions_df = pd.read_csv(Config.DATA_PATHS['interactions'])

clusters = cluster_all_users(
    users_df=users_df,
    reviews_df=reviews_df,
    interactions_df=interactions_df,
    n_clusters=5  # عدد المجموعات
)

print("تجميع المستخدمين في مجموعات")

# استخراج النتائج
print("عدد المستخدمين في كل مجموعة (مدمجة):")
for cluster_id, users in clusters["features_clusters"].items():
    print(f"المجموعة {cluster_id}: {len(users)} مستخدم")

print("=== تجميع الصفات ===")
print(clusters['features_clusters'].head(10))

print("\n=== تجميع التفاعلات ===")
print(clusters['interactions_clusters'].head(10))

print("\n=== التجميع المدمج ===")
print(clusters['combined_clusters'].head(10))

######################################################################################

## نظام التفسير
## نظام يولد تفسير لغوي للتوصيات التي تم اختيارها 
from explainer.explainer import RecommendationExplainer

# تحضير البيانات
user_data = {
    'preferences' : 'books'
}

product_data = {
    'title': 'ktabia',
    'category': 'books' ,
    'brand': 'dar alnasher'
}
explainer = RecommendationExplainer()
explanation = explainer.generate_explanation(
                user_data=user_data,
                product_data=product_data,
                recommendation_score=3
            )

print("توليد تفسير لغوي لتوصية معينة")
print(explanation)
print("################################################")

#####################################################################################

