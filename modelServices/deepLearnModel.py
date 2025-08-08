import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
from config.config import Config
from modelServices.data_processor import DataProcessor


class DeepLearnModel:
    def __init__(self):
        self.model_path= Config.DATA_PATHS['hybrid_model']
        self.model, self.processor, self.tokenizer = self.load_hybrid_model()


    def preprocess_data(self):
                users = pd.read_csv(Config.DATA_PATHS['users'])
                products = pd.read_csv(Config.DATA_PATHS['products'])
                interactions = pd.read_csv(Config.DATA_PATHS['interactions'])
                reviews = pd.read_csv(Config.DATA_PATHS['reviews'])
                context = pd.read_csv(Config.DATA_PATHS['context'])

                # دمج البيانات
                if 'interaction_id' not in interactions.columns:
                    interactions = interactions.reset_index().rename(columns={'index': 'interaction_id'})
                users = users.rename(columns={'location': 'location_user'})
                context = context.rename(columns={'location': 'location_con'})
                merged_data = pd.merge(interactions, context, on='interaction_id', how='left')
                merged_data = pd.merge(merged_data, users, on='user_id')
                merged_data = pd.merge(merged_data, products, on='product_id')
                merged_data = pd.merge(merged_data, reviews, on=['user_id', 'product_id'], how='left')

                # معالجة القيم المفقودة
                merged_data['rating'] = merged_data['rating'].fillna(merged_data['rating'].mean())
                merged_data['review_text'] = merged_data['review_text'].fillna('')

                # إعادة تعيين المعرفات لتبدأ من الصفر
                merged_data['user_id'] = LabelEncoder().fit_transform(merged_data['user_id'])
                merged_data['product_id'] = LabelEncoder().fit_transform(merged_data['product_id'])
                processor = DataProcessor()
                processor.fit(merged_data)
                import pickle
                processor_path = self.model_path.replace('.keras', '_processor.keras')
                with open(processor_path, 'wb') as f:
                        pickle.dump(processor, f)
                # print(f"تم حفظ معالج البيانات في: model_path/hybrid_model_processor.keras")
                # تحويل السمات الفئوية
                cat_cols = ['gender', 'location_user', 'category', 'brand', 'interaction_type',
                        'time_of_day', 'device', 'location_con']
                for col in cat_cols:
                    if col in merged_data.columns:
                        merged_data[col] = LabelEncoder().fit_transform(merged_data[col].astype(str))

                # تطبيع السمات الرقمية
                num_cols = ['age', 'price', 'rating']
                scaler = MinMaxScaler()
                merged_data[num_cols] = scaler.fit_transform(merged_data[num_cols])

                # معالجة النص مع ضبط حجم المفردات
                tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
                tokenizer.fit_on_texts(merged_data['review_text'])
                text_seq = tokenizer.texts_to_sequences(merged_data['review_text'])
                text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_seq, maxlen=100)
                #حفظ بيانات التدريب
                training_data_path = Config.DATA_PATHS['training_data_merged']
                merged_data.to_csv(training_data_path, index=False)
                return merged_data, text_padded, tokenizer

    def build_hybrid_model(self, num_users, num_products, vocab_size):
        # مدخلات النموذج
        user_id_input = Input(shape=(1,), name='user_id')
        product_id_input = Input(shape=(1,), name='product_id')
        user_features_input = Input(shape=(3,), name='user_features')
        product_features_input = Input(shape=(3,), name='product_features')
        context_input = Input(shape=(3,), name='context_features')
        text_input = Input(shape=(100,), name='text_input')

        # طبقات الـ Embedding مع حدود آمنة
        user_embedding = Embedding(input_dim=num_users, output_dim=64)(user_id_input)
        product_embedding = Embedding(input_dim=num_products, output_dim=64)(product_id_input)
        user_embedding = Flatten()(user_embedding)
        product_embedding = Flatten()(product_embedding)

        # طبقات كثيفة
        user_dense = Dense(32, activation='relu')(user_features_input)
        product_dense = Dense(32, activation='relu')(product_features_input)
        context_dense = Dense(32, activation='relu')(context_input)

        # معالجة النص
        text_embedding = Embedding(input_dim=vocab_size + 1, output_dim=64)(text_input)
        text_lstm = LSTM(32)(text_embedding)

        # دمج المميزات
        concat = Concatenate()([user_embedding, product_embedding,
                            user_dense, product_dense,
                            context_dense, text_lstm])

        dense = Dense(128, activation='relu')(concat)
        dense = Dropout(0.2)(dense)
        dense = Dense(64, activation='relu')(dense)
        output = Dense(1, activation='sigmoid')(dense)

        model = Model(inputs=[user_id_input, product_id_input,
                            user_features_input, product_features_input,
                            context_input, text_input],
                    outputs=output)

        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        return model

    def train_and_save_model(self):
            # معالجة البيانات
            data, text_padded, tokenizer = self.preprocess_data()
            # print(f"Max user_id: {data['user_id'].max()}, Num unique users: {data['user_id'].nunique()}")
            # print(f"Max product_id: {data['product_id'].max()}, Num unique products: {data['product_id'].nunique()}")

            # إعداد بيانات التدريب
            train_inputs = {
                'user_id': data['user_id'].values.reshape(-1, 1),
                'product_id': data['product_id'].values.reshape(-1, 1),
                'user_features': data[['age', 'gender', 'location_user']].values,
                'product_features': data[['category', 'price', 'brand']].values,
                'context_features': data[['time_of_day', 'device', 'location_con']].values,
                'text_input': text_padded
            }
            y_train = data['rating'].values

            # بناء النموذج
            num_users = data['user_id'].nunique()
            num_products = data['product_id'].nunique()
            vocab_size = len(tokenizer.word_index)
            model = self.build_hybrid_model(num_users, num_products, vocab_size)

            # تدريب النموذج
            history = model.fit(
                train_inputs, y_train,
                batch_size=64,
                epochs=50,
                validation_split=0.2,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=5),
                    tf.keras.callbacks.ModelCheckpoint(self.model_path, save_best_only=True)
                ]
            )

            # حفظ Tokenizer
            import pickle
            with open(f'{self.model_path}_tokenizer.keras', 'wb') as f:
                pickle.dump(tokenizer, f)

            return model, tokenizer,  history

    def load_hybrid_model(self):
        """
        تحميل النموذج مع جميع المكونات المساعدة
        """
        try:
            # تحميل النموذج
            model = keras.models.load_model(self.model_path, compile=False)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # تحميل tokenizer
            with open(f'{self.model_path}_tokenizer.keras', 'rb') as f:
                tokenizer = pickle.load(f)

            # تحميل معالج البيانات
            processor_path = self.model_path.replace('.keras', '_processor.keras')
            with open(processor_path, 'rb') as f:
                processor = pickle.load(f)

            # print("تم تحميل جميع المكونات بنجاح!")
            return model, processor, tokenizer

        except Exception as e:
            # print(f"فشل في التحميل: {str(e)}")
            return None, None, None

    def load_and_continue_training(self, epochs=10, batch_size=64):
            try:
                # 1. تحميل جميع المكونات
                # model, processor, tokenizer = self.load_hybrid_model(self.model_path)
                if self.model is None:
                    raise ValueError("فشل في تحميل المكونات الأساسية")


                new_data, text_padded, tokenizer = self.preprocess_data()
                # 6. البحث عن طبقات الـ Embedding بطريقة أكثر قوة
                def find_embedding_layer(model, layer_type):
                    for layer in model.layers:
                        if isinstance(layer, keras.layers.Embedding):
                            if layer_type == 'user' and layer.name.lower().startswith('user'):
                                return layer
                            elif layer_type == 'product' and layer.name.lower().startswith('product'):
                                return layer
                    return None

                user_embedding_layer = find_embedding_layer(self.model, 'user')
                product_embedding_layer = find_embedding_layer(self.model, 'product')

                # 7. البحث البديل إذا فشل البحث بالاسم
                if user_embedding_layer is None or product_embedding_layer is None:
                    embedding_layers = [layer for layer in self.model.layers
                                    if isinstance(layer, keras.layers.Embedding)]

                    if len(embedding_layers) >= 2:
                        user_embedding_layer, product_embedding_layer = embedding_layers[:2]
                    else:
                        # طباعة معلومات النموذج للمساعدة في التشخيص
                        # print("طبقات النموذج المتاحة:")
                        # for i, layer in enumerate(self.model.layers):
                        #     print(f"{i}: {layer.name} ({layer.__class__.__name__})")

                        raise ValueError("تعذر العثور على طبقات الـ Embedding المطلوبة")

                # 8. التحقق من نطاقات المعرفات
                max_user_id = user_embedding_layer.input_dim - 1
                max_product_id = product_embedding_layer.input_dim - 1

                # 9. تصفية البيانات
                valid_data = new_data[
                    (new_data['user_id'] <= max_user_id) &
                    (new_data['product_id'] <= max_product_id)
                ].copy()

                if len(valid_data) == 0:
                    raise ValueError("لا توجد بيانات صالحة للتدريب بعد التصفية")

                # 10. معالجة البيانات
                processed_data = self.processor.process_batch(valid_data)
                # processed_data=valid_data
                # 11. إعداد بيانات التدريب
                train_inputs = {
                        'user_id': processed_data['user_id'].values.reshape(-1, 1),
                        'product_id': processed_data['product_id'].values.reshape(-1, 1),
                        'user_features': processed_data[['age', 'gender', 'location_user']].values,
                        'product_features': processed_data[['category', 'price', 'brand']].values,
                        'context_features': processed_data[['time_of_day', 'device', 'location_con']].values,
                        'text_input': text_padded
                }
                y_train = processed_data['rating'].values

                # 12. استئناف التدريب
                history = self.model.fit(
                    train_inputs, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=[
                        keras.callbacks.EarlyStopping(patience=3),
                        keras.callbacks.ModelCheckpoint(self.model_path, save_best_only=True)
                    ]
                )

                # 13. الحفظ
                self.model.save(self.model_path)
                with open(f'{self.model_path}_processor.keras', 'wb') as f:
                    pickle.dump(self.processor, f)

                return self.model, history

            except Exception as e:
                # print(f"حدث خطأ أثناء استئناف التدريب: {str(e)}")
                return None, None

    def predict_rating(self, raw_input):
            """إصدار نهائي مع تتبع شامل للأخطاء"""
            try:
                # print("\n=== بدء عملية التنبؤ ===")

                # 1. التحقق من المدخلات الأساسية
                if not raw_input or not isinstance(raw_input, dict):
                    raise ValueError("بيانات الإدخال غير صالحة: يجب أن تكون قاموساً غير فارغ")

                # 2. معالجة القيم المفقودة
                # print("\nالبيانات قبل المعالجة:")
                # print(raw_input)

                raw_input = self.processor.handle_missing_values(raw_input)
                # print("\nالبيانات بعد معالجة القيم المفقودة:")
                # print(raw_input)

                # 3. معالجة البيانات الرئيسية
                processed_input = self.processor.process_input(raw_input)
                if processed_input is None:
                    raise ValueError("فشل في معالجة البيانات المدخلة")

                # print("\nالبيانات المعالجة:")
                # print(processed_input)

                # 4. معالجة النص
                if not processed_input.get('review_text', '').strip():
                    processed_input['review_text'] = "لا يوجد نص"
                    # print("تحذير: استخدام نص افتراضي")

                text_seq = self.tokenizer.texts_to_sequences([processed_input['review_text']])
                if not text_seq or not any(text_seq):
                    text_seq = [[1]]  # قيمة افتراضية

                text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_seq, maxlen=100)
                # print("\nالنص المعالج (الشكل):", text_padded.shape)

                # 5. إعداد مدخلات النموذج مع التحقق
                inputs = {
                    'user_id': np.array([processed_input['user_id']], dtype=np.int32),
                    'product_id': np.array([processed_input['product_id']], dtype=np.int32),
                    'user_features': np.array([[
                        processed_input.get('age', 0.5),
                        processed_input.get('gender', 0),
                        processed_input.get('location_user', 0)
                    ]], dtype=np.float32),
                    'product_features': np.array([[
                        processed_input.get('category', 0),
                        processed_input.get('price', 0.5),
                        processed_input.get('brand', 0)
                    ]], dtype=np.float32),
                    'context_features': np.array([[
                        processed_input.get('time_of_day', 0),
                        processed_input.get('device', 0),
                        processed_input.get('location_con', 0)
                    ]], dtype=np.float32),
                    'text_input': text_padded
                }

                # print("\nمدخلات النموذج:")
                # for k, v in inputs.items():
                #     print(f"{k}: {v.shape} {v.dtype}")

                # 6. التنبؤ
                # print("\nجاري التنبؤ...")
                prediction = self.model.predict(inputs, verbose=1)
                # print("نتيجة التنبؤ الخام:", prediction)
                prediction =prediction[0][0]
                return float(prediction)

            except Exception as e:
                # print(f"\n!!! فشل في التنبؤ: {str(e)}")
                import traceback
                traceback.print_exc()
                return None


