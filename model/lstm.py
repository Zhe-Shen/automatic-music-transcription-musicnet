import tensorflow as tf

number_layers = 3
number_classes = 128
number_units = 256

def getLSTM():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(number_units, return_sequences = "True",kernel_initializer='normal', activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.2))
    for i in range(number_layers - 1):
        model.add(tf.keras.layers.LSTM(number_units,return_sequences = "True",kernel_initializer='normal', activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(number_classes, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(5e-3), metrics=[
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5), 
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ])
    return model