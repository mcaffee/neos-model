import numpy as np
from sklearn.metrics import confusion_matrix
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import settings
from utils import plot_confusion_matrix


class NEOSModel:

    def __init__(self):
        # model_path = settings.BASE_SAVE_MODEL_PATH
        model_path = settings.BASE_CHECKPOINT_PATH
        print(f'Loading model: {model_path}')
        self.model: Model = load_model(model_path)

        # Test data
        ds_path = settings.DATA_DIR / 'input' / 'main-v1' / 'train'
        self.test_paths = {
            l: list((ds_path / l).glob('*.jpg'))
            for l in settings.CLASSES
        }
        datagen = ImageDataGenerator(
        )

        print(f'Loading dataset: {ds_path}')
        self.dataset = datagen.flow_from_directory(
            ds_path,
            target_size=settings.IMAGE_SIZE,
            batch_size=1,
            classes=settings.CLASSES,
            shuffle=False,
        )

        # self.model.summary()

    def validate(self):
        self.model.evaluate(self.dataset)
        self.print_confusion_matrix()
        # self.print_stats()

    def predict(self, path):
        img = self.load_image(path)
        preds = self.model.predict(img)[0]
        preds = [round(float(v), 2) for v in preds]
        preds_dict = dict(zip(settings.CLASSES, preds))
        return preds_dict

    def print_confusion_matrix(self):
        y_pred = self.model.predict(self.dataset)
        y_pred = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(self.dataset.labels, y_pred)
        plot_confusion_matrix(cm)

    def print_stats(self):
        letter_precisions = []
        for letter, paths in self.test_paths.items():
            letter_preds = [
                self.predict(path)[letter]
                for path in paths
            ]

            letter_precision = round(sum(letter_preds) / len(letter_preds), 2)
            letter_precisions.append(letter_precision)
            print(
                f'{letter}: {letter_precision}, max {round(max(letter_preds), 2)}, '
                f'min {round(min(letter_preds), 2)}, {letter_preds}'
            )

        print(f'Model total precision: {round(sum(letter_precisions) / len(settings.CLASSES), 2)}')

    def load_image(self, path):
        img = image.load_img(path, target_size=settings.IMAGE_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return x


if __name__ == '__main__':
    model = NEOSModel()
    model.validate()

    # for p in model.test_paths['А'][0]:
    #     print(model.predict(p))

    # p = '/Users/a842799yara.com/Sync/Users/mcaffee/Projects/itmsc/05 kurs 2 semester 2/vkr/project/model-training/data/input/main/train/Е/IMG_0001.JPG_ench.jpg'
    # print(model.predict(p))
