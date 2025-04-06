from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import logging
import io
from PIL import Image

class RiceDiseaseClassifier:
    CLASS_MAPPING = {
        0: 'Bacterial Blight',
        1: 'Brown Spot',
        2: 'Healthy',
        3: 'Hispa',
        4: 'Leaf Blast',
        5: 'Sheath Blight',
        6: 'Tungro'
    }

    def __init__(self):
        self.logger = self._setup_logging()
        self.models_list = self._load_individual_models()
        self.ensemble_weights = np.array([0.299625468164794, 0.3445692883895131, 0.35580524344569286], dtype=np.float32)
        self.ensemble_model = self._build_ensemble_model()

    @staticmethod
    def _setup_logging():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        return logging.getLogger(__name__)

    def _load_individual_models(self):
        try:
            models = [
                load_model('/Users/vanhoklin/Desktop/model/densenet121_7classes_model.h5'),
                load_model('/Users/vanhoklin/Desktop/model/mobilenetv2_7classes_model.h5'),
                self._create_efficientnet_model()
            ]
            self.logger.info("Individual models loaded successfully")
            return models
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

    def _create_efficientnet_model(self):
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(7, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights('/Users/vanhoklin/Desktop/model/efficientnetb0_weights.h5')
        return model

    def _build_ensemble_model(self):
        def ensemble_prediction(inputs):
            predictions = [model.predict(inputs, verbose=0) for model in self.models_list]
            return np.average(predictions, weights=self.ensemble_weights, axis=0)

        return ensemble_prediction

    def preprocess_image(self, file):
        try:
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            image = image.resize((224, 224))
            image_array = np.array(image, dtype=np.float32) / 255.0
            return np.expand_dims(image_array, axis=0)
        except Exception as e:
            self.logger.error(f"Image preprocessing error: {e}")
            raise

    def predict(self, image):
        try:
            predictions = self.ensemble_model(image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = float(predictions[0][predicted_class])

            self.logger.info(f"Prediction: class={predicted_class}, confidence={confidence}")
            return {
                'class': self.CLASS_MAPPING[predicted_class],
                'confidence': confidence
            }
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise

app = Flask(__name__)
classifier = RiceDiseaseClassifier()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    try:
        image = classifier.preprocess_image(request.files['file'])
        result = classifier.predict(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Welcome to the Rice Disease Classification API!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)