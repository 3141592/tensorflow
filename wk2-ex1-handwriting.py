import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils.vis_utils import plot_model

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss') < 0.01):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, tf.nn.relu),
                                    tf.keras.layers.Dense(10, tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=999, callbacks=[callbacks])
model.evaluate(x_test, y_test)

classifications = model.predict(x_test)


print(classifications[1])
print(y_test[1])

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

