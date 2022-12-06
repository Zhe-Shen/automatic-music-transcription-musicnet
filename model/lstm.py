import tensorflow as tf
from types import SimpleNamespace

WINDOW_SIZE = 64

class MemoryRNN(tf.keras.Model):

    ##########################################################################################

    def __init__(self, note_size=128, rnn_size=256):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param rnn_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()

        self.note_size = note_size
        self.rnn_size = rnn_size

        ## TODO: Finish off the method as necessary.
        ## Define an embedding component to embed the word indices into a trainable embedding space.
        ## Define a recurrent component to reason with the sequence of data. 
        ## You may also want a dense layer near the end...
        self.model = tf.keras.Sequential([
            tf.keras.layers.GRU(units=rnn_size, return_sequences=True),
            tf.keras.layers.Dense(units=note_size, activation='sigmoid')
        ])    

    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.
        """
        ## TODO: Implement the method as necessary
        # inputs = tf.reshape(inputs, (-1, WINDOW_SIZE))
        return self.model(inputs)
    
    
def get_text_model():
    '''
    Tell our autograder how to train and test your model!
    '''

    ## TODO: Set up your implementation of the RNN

    ## Optional: Feel free to change or add more arguments!
    model = MemoryRNN()

    ## TODO: Define your own loss and metric for your optimizer
    loss_metric = "mse"

    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), 
        loss=loss_metric, 
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5), 
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ],
    )

    return SimpleNamespace(
        model = model,
        epochs = 10,
        batch_size = 100,
    )