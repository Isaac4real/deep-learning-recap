import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, Flatten, Dense


def basic_nn():
    # https://www.tensorflow.org/tutorials/quickstart/beginner
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # flattens the image to (,784)
        tf.keras.layers.Dense(units=128, activation="relu"),
        # pure NN that outputs activation(dot(input, weights) + bias); weights is (784,); weights are assigned by tf
        tf.keras.layers.Dropout(rate=0.2),  # set values to 0 at rate 0.2, scales the others by 1/0.8
        tf.keras.layers.Dense(units=10)
    ])
    # the above is a classic NN; the ideal would be a image_classification

    # the NN/model/function is defined, but now we need to defined how we optimize that NN/function's parameters such that we find the minimum loss...aka, let's define our loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)  # from_logits=True since the model output is not softmax

    model.compile(optimizer="adam",
                  loss=loss_fn,
                  metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)


def cnn():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=30, kernel_size=3, input_shape=(28, 28, 1), activation="relu"),
        # cross-correlates throughout the input image. could test flatteing the input first
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Conv2D(filters=60, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=60),
        tf.keras.layers.Dense(units=10)
    ])
    # the above is a classic NN; the ideal would be a image_classification

    # the NN/model/function is defined, but now we need to defined how we optimize that NN/function's parameters such that we find the minimum loss...aka, let's define our loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)  # from_logits=True since the model output is not softmax

    model.compile(optimizer="adam",
                  loss=loss_fn,
                  metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


def advanced_cnn():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    model = MyModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )


if __name__ == "__main__":
    # basic_nn() # loss: 0.0759 - accuracy: 0.9763 - 428ms/epoch - 1ms/step
    # cnn()  # loss: 0.0397 - accuracy: 0.9879 - 2s/epoch - 6ms/step
    advanced_cnn()  # Epoch 5, Loss: 0.00917352270334959, Accuracy: 99.70166778564453, Test Loss: 0.05794723331928253, Test Accuracy: 98.3499984741211

