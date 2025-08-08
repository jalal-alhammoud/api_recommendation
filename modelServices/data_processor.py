from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import logging

class DataProcessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.tokenizer = None
        self.default_values = {}  # لتخزين القيم الافتراضية
        self.logger = logging.getLogger(__name__)

    def fit(self, data):
        """تدريب معالجات البيانات مع تسجيل القيم الافتراضية"""
        # المعالجة الرقمية
        num_cols = ['age', 'price']
        for col in num_cols:
            self.scalers[col] = MinMaxScaler()
            self.scalers[col].fit(data[[col]])
            self.default_values[col] = data[col].median()  # تخزين الوسيط كقيمة افتراضية

        # المعالجة الفئوية
        cat_cols = ['gender', 'location_user', 'category', 'brand',
                   'time_of_day', 'device', 'location_con']
        for col in cat_cols:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(data[col].astype(str))
            # تخزين القيمة الأكثر تكرارا كافتراضية
            self.default_values[col] = data[col].mode()[0]

    def safe_transform(self, encoder, values, col_name):
        """تحويل آمن للبيانات الفئوية مع التعامل مع القيم الجديدة"""
        try:
            return encoder.transform(values)
        except ValueError:
            # إذا كانت القيمة جديدة، نستخدم قيمة افتراضية
            default_val = self.default_values.get(col_name, 'Unknown')
            # print(f"تحذير: قيمة غير معروفة في {col_name}، استخدام القيمة الافتراضية: {default_val}")
            return encoder.transform([default_val] * len(values))

    def process_input(self, raw_input):
          """إصدار معدل مع تحسينات تتبع الأخطاء"""
          try:
              #   print("بيانات الإدخال الخام:", raw_input)  # للتتبع

              # 1. التحقق من وجود الحقول الأساسية
              required_fields = ['user_id', 'product_id', 'age', 'gender',
                              'price', 'review_text']
              for field in required_fields:
                  if field not in raw_input:
                      self.logger.error(f"الحقل المطلوب {field} مفقود")
                      raise

              # 2. إنشاء قاموس المعالجة
              processed = {
                  'user_id': int(raw_input['user_id']),
                  'product_id': int(raw_input['product_id']),
                  'review_text': str(raw_input['review_text'])
              }

              # 3. المعالجة الرقمية مع التحقق
              if 'age' in self.scalers:
                  processed['age'] = self.scalers['age'].transform([[float(raw_input['age'])]])[0][0]
              else:
                  raise ValueError("Scaler للعمر غير موجود")

              if 'price' in self.scalers:
                  processed['price'] = self.scalers['price'].transform([[float(raw_input['price'])]])[0][0]
              else:
                  raise ValueError("Scaler للسعر غير موجود")

              # 4. المعالجة الفئوية مع التحقق
              categorical_mapping = {
                  'gender': 'gender',
                  'location_user': 'location_x',
                  'category': 'category',
                  'brand': 'brand',
                  'time_of_day': 'time_of_day',
                  'device': 'device',
                  'location_con': 'location_y'
              }

              for input_field, processor_field in categorical_mapping.items():
                  if input_field in raw_input and processor_field in self.encoders:
                      processed[input_field] = self.encoders[processor_field].transform(
                          [str(raw_input[input_field])]
                      )[0]
                  else:
                    #   print(f"تحذير: حقل {input_field} غير موجود أو لا يوجد معالج له")
                      processed[input_field] = 0  # قيمة افتراضية

            #   print("البيانات المعالجة بنجاح:", processed)
              return processed

          except Exception as e:
            #   print(f"خطأ في معالجة المدخلات: {str(e)}")
              import traceback
              traceback.print_exc()
              return None
    def process_batch(self, batch_data):
        """معالجة دفعة كاملة مع التعامل مع القيم الجديدة"""
        processed = batch_data.copy()

        # المعالجة الرقمية
        for col in ['age', 'price']:
            if col in self.scalers:
                # استبدال القيم المفقودة بالقيمة الافتراضية
                processed[col] = processed[col].fillna(self.default_values[col])
                processed[col] = self.scalers[col].transform(processed[[col]])

        # المعالجة الفئوية
        for col, encoder in self.encoders.items():
            if col in processed.columns:
                # تحويل آمن مع التعامل مع القيم الجديدة
                processed[col] = processed[col].astype(str).fillna(str(self.default_values[col]))
                processed[col] = self.safe_transform(
                    encoder,
                    processed[col],
                    col
                )

        return processed

    def handle_missing_values(self, raw_input):
            """معالجة القيم المفقودة في البيانات"""
            # defaults = {
            #     'age': 30,  # متوسط العمر المتوقع
            #     'gender': 'غير محدد',
            #     'price': 0,
            #     'review_text': ''
            # }

            for key in self.default_values:
                if key not in raw_input or raw_input[key] is None:
                    raw_input[key] = self.default_values[key]
            if 'user_id' not in raw_input or raw_input['user_id'] is None:
                raw_input['user_id'] = 1
            if 'product_id' not in raw_input or raw_input['product_id'] is None:
                raw_input['product_id'] = 1

            return raw_input
    def validate_input(self, raw_input):
          """التحقق من صحة البيانات المدخلة"""
          required = {
              'user_id': int,
              'product_id': int,
              'age': (int, float),
              'gender': str,
              'price': (int, float),
              'review_text': str
          }

          for field, types in required.items():
              if field not in raw_input:
                  return False, f"الحقل {field} مفقود"
              if not isinstance(raw_input[field], types):
                  return False, f"الحقل {field} يجب أن يكون من نوع {types}"

          return True, "البيانات صالحة"

    def get_default_values(self):
            """الحصول على القيم الافتراضية للاستخدام عند وجود أخطاء"""
            return {
                'user_id': 0,
                'product_id': 0,
                'age': 30,
                'gender': 'unknown',
                'price': 0,
                'review_text': '',
                'category': 'other',
                'brand': 'unknown',
                'time_of_day': 'day',
                'device': 'desktop',
                'location_user': 'unknown',
                'location_con': 'unknown'
            }

    def check_processor(processor):
              """فحص حالة المعالج قبل الاستخدام"""
              print("\n=== فحص المعالج ===")

              # 1. التحقق من السكالرز
              if not hasattr(processor, 'scalers'):
                  print("تحذير: لا يوجد scalers في المعالج")
              else:
                  print("Scalers المتاحة:", list(processor.scalers.keys()))

              # 2. التحقق من الانكودرز
              if not hasattr(processor, 'encoders'):
                  print("تحذير: لا يوجد encoders في المعالج")
              else:
                  print("Encoders المتاحة:", list(processor.encoders.keys()))

              # 3. التحقق من القيم الافتراضية
              if not hasattr(processor, 'default_values'):
                  print("تحذير: لا يوجد default_values في المعالج")
              else:
                  print("بعض القيم الافتراضية:", {k: processor.default_values[k] for k in list(processor.default_values.keys())[:3]})

