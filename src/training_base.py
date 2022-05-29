from typing import Tuple, Optional

import numpy as np

import settings

if settings.MONITORING:
    from comet_ml import Experiment

from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.applications.mobilenet import preprocess_input, MobileNet
from keras import Input
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.metrics import Precision, Recall
from keras.optimizer_v2.adam import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

from utils import plot_confusion_matrix, visualize_ds
from callbacks import ConfusionMatrixCallback


class BaseModel:

    def __init__(self):
        self.base_model = MobileNet(
            weights='imagenet',
            input_shape=(*settings.IMAGE_SIZE, 3),
            include_top=False
        )

        inputs = Input(shape=(*settings.IMAGE_SIZE, 3), dtype=tf.uint8)
        x = tf.cast(inputs, tf.float32)
        x = preprocess_input(x)
        x = self.base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(len(settings.CLASSES), activation='softmax')(x)

        self.model: Model = Model(inputs=[inputs], outputs=[outputs])

        # Datasets
        self.train_ds, self.validation_ds, self.test_ds = self.load_datasets()

        # Monitoring
        if settings.MONITORING:
            self.experiment = Experiment(
                api_key=settings.COMET_API_KEY,
                project_name='ictis-neos',
                workspace='mcaffee',
            )
            self.experiment.log_parameters(
                {
                    'image_size': '{}x{}'.format(*settings.IMAGE_SIZE),
                    'training_batch_size': settings.BASE_TRAINING_NBATCH,
                    'training_epochs': settings.BASE_TRAINING_EPOCHS,
                    'validation_batch_size': settings.BASE_VALIDATION_NBATCH,
                    'fine_tuning_epochs': settings.BASE_TRAINING_FINE_TUNING_EPOCHS,
                    'classes': str(settings.CLASSES),
                    'checkpoint_path': settings.BASE_CHECKPOINT_PATH,
                    'save_model_path': settings.BASE_SAVE_MODEL_PATH,
                    'train_data_path': settings.BASE_TRAIN_DATA_PATH,
                    'test_data_path': settings.BASE_TEST_DATA_PATH,
                }
            )
            self.experiment.add_tags(['main-v3', 'transfer'])

        # Callbacks
        self.callbacks = [
            ModelCheckpoint(
                filepath=settings.BASE_CHECKPOINT_PATH,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1,
            ),
        ]
        if settings.MONITORING and settings.LOG_CONFUSION_MATRIX:
            self.callbacks.append(
                ConfusionMatrixCallback(
                    experiment=self.experiment,
                    dataset=self.test_ds,
                )
            )

    def train(self):
        print('train: Training...')
        self.lock()

        self.model.compile(
            optimizer=Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )

        self.model.fit(
            self.train_ds,
            epochs=settings.BASE_TRAINING_EPOCHS,
            validation_data=self.validation_ds,
            callbacks=self.callbacks,
        )

    def fine_tune(self):
        print('fine_tune: Fine-Tuning...')
        self.unlock()
        self.model.compile(
            optimizer=Adam(1e-5),  # Very low learning rate
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )

        self.model.fit(
            self.train_ds,
            epochs=settings.BASE_TRAINING_FINE_TUNING_EPOCHS,
            callbacks=self.callbacks,
            validation_data=self.validation_ds,
        )

    def validate(self):
        print('validate: Validating...')
        y_pred = self.model.predict(self.test_ds)
        y_pred = np.argmax(y_pred, axis=1)

        if settings.MONITORING:
            self.experiment.log_confusion_matrix(
                self.test_ds.labels,
                y_pred,
                title='Final Confusion Matrix, Test Dataset',
                file_name='confusion-matrix-final-test.json',
            )

        else:
            cm = confusion_matrix(self.test_ds.labels, y_pred)
            plot_confusion_matrix(cm)

    def save(self):
        print('save: Saving...')
        self.model.save(settings.BASE_SAVE_MODEL_PATH)

    def lock(self, num_layers: Optional[int] = None):
        print('lock: Locking...')
        if num_layers is not None:
            index = len(self.base_model.layers) - num_layers

            for layer in self.base_model.layers[:index]:
                layer.trainable = False
            for layer in self.base_model.layers[index:]:
                layer.trainable = True
        else:
            for layer in self.base_model.layers:
                layer.trainable = False

    def unlock(self):
        print('unlock: Unlocking...')
        for layer in self.base_model.layers:
            layer.trainable = True

    def load_datasets(self) -> Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]:
        print(f'load_datasets: Loading normal from {settings.BASE_TRAIN_DATA_PATH}')
        train_datagen = ImageDataGenerator(
            rotation_range=12.,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.15,
            shear_range=0.2,
            brightness_range=(0.5, 1.0),
            horizontal_flip=True,
        )
        validate_datagen = ImageDataGenerator(
        )

        train_ds = train_datagen.flow_from_directory(
            settings.BASE_TRAIN_DATA_PATH,
            target_size=settings.IMAGE_SIZE,
            batch_size=settings.BASE_TRAINING_NBATCH,
            classes=settings.CLASSES,
        )
        validate_ds = validate_datagen.flow_from_directory(
            settings.BASE_VALIDATE_DATA_PATH,
            target_size=settings.IMAGE_SIZE,
            batch_size=settings.BASE_VALIDATION_NBATCH,
            classes=settings.CLASSES,
            shuffle=False,
        )
        test_ds = validate_datagen.flow_from_directory(
            settings.BASE_TEST_DATA_PATH,
            target_size=settings.IMAGE_SIZE,
            batch_size=1,
            classes=settings.CLASSES,
            shuffle=False,
        )

        visualize_ds(train_ds)
        return train_ds, validate_ds, test_ds


if __name__ == '__main__':
    model = BaseModel()
    model.train()
    model.fine_tune()
    model.save()
    model.validate()
