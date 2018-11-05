# TensorFlow and tf.keras
import argparse

import tensorflow as tf
import trainer.model
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        default='/tmp/census-estimator')
    args, _ = parser.parse_known_args()

    print(tf.__version__)

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    model = trainer.model.getModel()
    model.fit(train_images, train_labels, epochs=5)
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

