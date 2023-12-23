import numpy as np
import tensorflow as tf
import sys
import os



# Get the parent directory
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
 
# setting path
# sys.path.append(directory.parent.parent)
 

from BaseAttention import BaseAttention
from PositonalEmbedding import PositionalEmbedding



class CausalSelfAttention(BaseAttention):
    """
    Call self attention on the input sequence, ensuring that each position in the 
    output depends only on previous positions (i.e. a causal model).

    Methods:
        call: Performs the forward pass of the layer.

    Attributes:
        mha (tf.keras.layers.MultiHeadAttention): The MultiHeadAttention layer.
        layernorm (tf.keras.layers.LayerNormalization): The LayerNormalization layer.
        add (tf.keras.layers.Add): The Add layer.
    """
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        The call function that performs the causal self-attention operation.
        
        Args:
            x (tf.Tensor): The input sequence of shape (batch_size, seq_length, d_model).

        Returns:
            tf.Tensor: The output sequence of shape (batch_size, seq_length, d_model).
        """
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    

if __name__ == "__main__":
    decoder_vocab_size = 1100
    d_model = 512

    decoder_embedding_layer = PositionalEmbedding(decoder_vocab_size, d_model)

    random_decoder_input = np.random.randint(0, decoder_vocab_size, size=(1, 110))

    decoder_embeddings = decoder_embedding_layer(random_decoder_input)

    print("decoder_embeddings shape", decoder_embeddings.shape)

    causal_self_attention_layer = CausalSelfAttention(num_heads=2, key_dim=512)
    causal_self_attention_output = causal_self_attention_layer(decoder_embeddings)

    print("causal_self_attention_output shape", causal_self_attention_output.shape)

    out1 = causal_self_attention_layer(decoder_embedding_layer(random_decoder_input[:, :50])) # Only the first 50 tokens beffore applying the embedding layer
    out2 = causal_self_attention_layer(decoder_embedding_layer(random_decoder_input)[:, :50]) # Only the first 50 tokens after applying the embedding layer

    diff = tf.reduce_max(tf.abs(out1 - out2)).numpy()

    print("Difference between the two outputs:", diff)