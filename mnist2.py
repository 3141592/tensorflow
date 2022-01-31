import tensorflow as tf
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs,:,:], y_data[idxs]

# Python optimisation variables
epochs = 10
batch_size = 100

# normalize the input images by dividing by 255.0
x_train = x_train / 255.0
x_test = x_test / 255.0
# convert x_test to tensor to pass through model (train data will be converted to
# tensors on the fly)
x_test = tf.Variable(x_test)

print(x_test)

