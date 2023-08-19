import tensorflow as tf
from utils import positional_encoding
from encoderLayer import EncoderLayer

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff,
               maximum_position_encoding, rate=0.1, n_features=None):
    super(Encoder, self).__init__()
    self.inp = tf.keras.layers.InputLayer(input_shape=(maximum_position_encoding, n_features))

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Conv1D(d_model,1)
    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.d_model)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask=None):

    seq_len = tf.shape(x)[1]
    x = self.inp(x)

    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    # x += self.pos_encoding_trainbale[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)
