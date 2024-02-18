import os 
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
from dataclasses import dataclass
from pathlib import Path 
from src.cnnClassifier.entity.config_entity  import PrepareBaseModelConfig
   
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        model = VGG16()
        # Restructure the model
        self.model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        self.save_model(path=self.config.base_model_path, model=self.model)

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
            max_length=self.config.params_max_length,
            vocab_size=self.config.params_vocab_size
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    @staticmethod
    def _prepare_full_model(model, freeze_all, freeze_till, learning_rate, max_length,vocab_size):
        # Define the model architecture here
        inputs1 = Input(shape=(4096,))
        fe1 = Dropout(0.4)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.4)(se1)
        se3 = LSTM(256)(se2)
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.summary()
        return model
