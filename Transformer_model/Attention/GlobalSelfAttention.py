import tensorflow as tf
import sys
import os
# Get the parent directory
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from BaseAttention import BaseAttention


class GlobalSelfAttention(BaseAttention):
    """
    A class that implements the global self-attention layer by inheriting from the BaseAttention class.
    This layer is used to process a single sequence and attends to all the tokens in the sequence.

    Methods:
        call: Performs the forward pass of the layer.

    Attributes:
        mha (tf.keras.layers.MultiHeadAttention): The MultiHeadAttention layer.
        layernorm (tf.keras.layers.LayerNormalization): The LayerNormalization layer.
        add (tf.keras.layers.Add): The Add layer.
    """
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        The call function that performs the global self-attention operation.

        Args:
            x (tf.Tensor): The input sequence of shape (batch_size, seq_length, d_model).

        Returns:
            tf.Tensor: The output sequence of shape (batch_size, seq_length, d_model).
        """
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
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

    cross_attention_layer = GlobalSelfAttention(num_heads=2, key_dim=512)
    cross_attention_output = cross_attention_layer(encoder_embeddings)

    print("global_self_attention_output shape", cross_attention_output.shape)