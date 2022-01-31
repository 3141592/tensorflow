import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils.vis_utils import plot_model

mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(training_images[42])

training_images = training_images / 255
test_images = test_images / 255

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, tf.nn.relu),
                                    tf.keras.layers.Dense(10, tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=20)
model.evaluate(test_images, test_labels)
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
