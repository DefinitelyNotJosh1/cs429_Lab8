# tutorial - 1
# Python 3.11.9
# Tensorflow2 quickstart for beginners
# https://www.tensorflow.org/tutorials/quickstart/beginner

import tensorflow as tf
print("Tensorflow version:", tf.__version__)

minist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = minist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print(predictions)

tf.nn.softmax(predictions).numpy()
print(tf.nn.softmax(predictions).numpy())

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

print(loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

evaluation = model.evaluate(x_test,  y_test, verbose=2)

print(evaluation)