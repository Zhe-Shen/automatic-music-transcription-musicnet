import tensorflow as tf

def getDNN():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1024, activation='leaky_relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512, activation='leaky_relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(5e-3), metrics=[
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5), 
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ])
    return model