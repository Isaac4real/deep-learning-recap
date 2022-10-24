''' https://www.tensorflow.org/tutorials/keras/text_classification '''
if __name__ == "__main__":
    import re
    import shutil
    import string

    import tensorflow as tf

    ### 1. load data
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                      untar=True, cache_dir='.',
                                      cache_subdir='')

    shutil.rmtree('aclImdb/train/unsup')  # only pos/ and neg/ are needed
    # using text_dataset_from_directory from tf.data to split data into train, validation and test
    batch_size = 32
    seed = 42
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)

    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/test',
        batch_size=batch_size)


    ### 2. Prepare data for training:
    #       a. standardize (preprocess data: remove punctuation, etc)
    #       b. tokenize (e.g. splitting sentences into individual words by splitting on whitespace)
    #       c. vectorize (convert tokens into vectors, so that they can be fed into a NN)

    # To prevent training-testing skew (also known as training-serving skew), it is important to preprocess the data identically at train and test time. To facilitate this, the TextVectorization layer can be included directly inside your model, as shown later in this tutorial.
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation),
                                        '')


    max_features = 10000
    sequence_length = 250
    # It's important to only use your training data when calling adapt (using the test set would leak information).
    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)


    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label


    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    # optimize dataset performance, such that I/O is not a bottleneck
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    ##  3. model creation
    embedding_dim = 16
    model = tf.keras.Sequential([
      tf.keras.layers.Embedding(max_features + 1, embedding_dim),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.GlobalAveragePooling1D(),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(1)])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    ## 4. Train
    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)

    ## 5. evaluate
    loss, accuracy = model.evaluate(test_ds)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    ## 6. create a new model based on the weights of the above model that incorporates the TextVectorization layer into it. This model is then capable of processing raw stinrgs (e.g to simplify deploying it)
    export_model = tf.keras.Sequential([
      vectorize_layer,
      model,
      tf.keras.layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    # Test it with `raw_test_ds`, which yields raw strings
    loss, accuracy = export_model.evaluate(raw_test_ds)
    print(accuracy)

    examples = [
      "The movie was great!",
      "The movie was okay.",
      "The movie was terrible..."
    ]

    export_model.predict(examples)
