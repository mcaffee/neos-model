import tensorflow as tf

import settings

if __name__ == '__main__':
    converter = tf.lite.TFLiteConverter.from_saved_model(str(settings.USER_SAVE_MODEL_PATH))

    tflite_model = converter.convert()
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    with open(settings.DATA_DIR / 'output' / 'model-user.tflite', 'wb') as outfile:
        outfile.write(tflite_model)
