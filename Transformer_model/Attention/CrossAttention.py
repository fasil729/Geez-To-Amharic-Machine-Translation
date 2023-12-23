import tensorflow as tf
import sys
import os
# Get the parent directory
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from BaseAttention import BaseAttention

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class CrossAttention(BaseAttention):
    """
    A class that implements the cross-attention layer by inheriting from the BaseAttention class.
    This layer is used to process two different sequences and attends to the context sequence while processing the query sequence.

    Methods:
        call: Performs the forward pass of the layer.    

    Attributes:
        mha (tf.keras.layers.MultiHeadAttention): The MultiHeadAttention layer.
        layernorm (tf.keras.layers.LayerNormalization): The LayerNormalization layer.
        add (tf.keras.layers.Add): The Add layer.
    """
    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """
        The call function that performs the cross-attention operation.

        Args:
            x (tf.Tensor): The query (expected Transformer results) sequence of shape (batch_size, seq_length, d_model).
            context (tf.Tensor): The context (inputs to the Encoder layer) sequence of shape (batch_size, seq_length, d_model).

        Returns:
            tf.Tensor: The output sequence of shape (batch_size, seq_length, d_model).
        """
        attn_output, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x



if __name__ == "__main__":
    import numpy as np
    from PositonalEmbedding import PositionalEmbedding  

    encoder_vocab_size = 1000
    
    decoder_vocab_size = 1100
    d_model = 512

    encoder_embedding_layer = PositionalEmbedding(encoder_vocab_size, d_model)
    decoder_embedding_layer = PositionalEmbedding(decoder_vocab_size, d_model)

    random_encoder_input = np.random.randint(0, encoder_vocab_size, size=(1, 100))
    random_decoder_input = np.random.randint(0, decoder_vocab_size, size=(1, 110))

    encoder_embeddings = encoder_embedding_layer(random_encoder_input)
    decoder_embeddings = decoder_embedding_layer(random_decoder_input)

    print("encoder_embeddings shape", encoder_embeddings.shape)
    print("decoder_embeddings shape", decoder_embeddings.shape)

    cross_attention_layer = CrossAttention(num_heads=2, key_dim=512)
    cross_attention_output = cross_attention_layer(decoder_embeddings, encoder_embeddings)

    print("cross_attention_output shape", cross_attention_output.shape)
