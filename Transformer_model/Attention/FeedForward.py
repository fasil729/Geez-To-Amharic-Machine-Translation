import tensorflow as tf
import sys
import os
# Get the parent directory
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)





class FeedForward(tf.keras.layers.Layer):
    """
    A class that implements the feed-forward layer.

    Methods:
        call: Performs the forward pass of the layer.

    Attributes:
        seq (tf.keras.Sequential): The sequential layer that contains the feed-forward layers. It applies the two feed-forward layers and the dropout layer.
        add (tf.keras.layers.Add): The Add layer.
        layer_norm (tf.keras.layers.LayerNormalization): The LayerNormalization layer.
    """
    def __init__(self, d_model: int, dff: int, dropout_rate: float=0.1):
        """
        Constructor of the FeedForward layer.

        Args:
            d_model (int): The dimensionality of the model.
            dff (int): The dimensionality of the feed-forward layer.
            dropout_rate (float): The dropout rate.
        """
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        The call function that performs the feed-forward operation. 

        Args:
            x (tf.Tensor): The input sequence of shape (batch_size, seq_length, d_model).

        Returns:
            tf.Tensor: The output sequence of shape (batch_size, seq_length, d_model).
        """
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x
    


if __name__ == "__main__":
    import numpy as np
    from PositonalEmbedding import PositionalEmbedding

    encoder_vocab_size = 1000
    d_model = 512
    

    encoder_embedding_layer = PositionalEmbedding(encoder_vocab_size, d_model)

    random_encoder_input = np.random.randint(0, encoder_vocab_size, size=(1, 100))

    encoder_embeddings = encoder_embedding_layer(random_encoder_input)

    print("encoder_embeddings shape", encoder_embeddings.shape)

    feed_forward_layer = FeedForward(d_model, dff=2048)
    feed_forward_output = feed_forward_layer(encoder_embeddings)

    print("feed_forward_output shape", feed_forward_output.shape)
