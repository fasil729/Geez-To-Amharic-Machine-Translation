import tensorflow as tf
import sys
import os
# Get the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_dir)

from Transformer_model.Attention.GlobalSelfAttention import GlobalSelfAttention
from Transformer_model.Attention.FeedForward import FeedForward


class EncoderLayer(tf.keras.layers.Layer):
    """
    A single layer of the Encoder. Usually there are multiple layers stacked on top of each other.

    Methods:
        call: Performs the forward pass of the layer.

    Attributes:
        self_attention (GlobalSelfAttention): The global self-attention layer.
        ffn (FeedForward): The feed-forward layer.
    """
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float=0.1):
        """
        Constructor of the EncoderLayer.

        Args:
            d_model (int): The dimensionality of the model.
            num_heads (int): The number of heads in the multi-head attention layer.
            dff (int): The dimensionality of the feed-forward layer.
            dropout_rate (float): The dropout rate.
        """
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
            )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        The call function that performs the forward pass of the layer.

        Args:
            x (tf.Tensor): The input sequence of shape (batch_size, seq_length, d_model).

        Returns:
            tf.Tensor: The output sequence of shape (batch_size, seq_length, d_model).
        """
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
    


if __name__ == "__main__":
    from Transformer_model.PositonalEmbedding import PositionalEmbedding
    import numpy as np

    encoder_vocab_size = 1000
    d_model = 512

    encoder_embedding_layer = PositionalEmbedding(encoder_vocab_size, d_model)

    random_encoder_input = np.random.randint(0, encoder_vocab_size, size=(1, 100))

    encoder_embeddings = encoder_embedding_layer(random_encoder_input)

    print("encoder_embeddings shape", encoder_embeddings.shape)

    encoder_layer = EncoderLayer(d_model, num_heads=2, dff=2048)

    encoder_layer_output = encoder_layer(encoder_embeddings)

    print("encoder_layer_output shape", encoder_layer_output.shape)