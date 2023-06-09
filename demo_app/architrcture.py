import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()

        # defining  layers
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        
        # Transformer block
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # dense layer
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        # data prepaation
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        
        # Embedding
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        
        # concatenation
        embeddings = token_embeddings + position_embeddings
        return embeddings


class NERModel(keras.Model):
    def __init__(
        self, num_tags, vocab_size, maxlen=128, embed_dim=32, num_heads=2, ff_dim=32
    ):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = layers.Dropout(0.1)
        self.ff = layers.Dense(ff_dim, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.ff_final = layers.Dense(num_tags, activation="softmax")

    def call(self, inputs, training=False):
        
        # Block-1 (Feature extraction)
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)

        # Fully Connected
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x


class CustomNonPaddingTokenLoss(keras.losses.Loss):
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # calculating the loss directly
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=keras.losses.Reduction.NONE
        )
        loss = loss_fn(y_true, y_pred)

        # filtering the masked tags
        mask = tf.cast((y_true > 0), dtype=tf.float32)
        loss = loss * mask

        # finging out the ratio
        ratio = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return ratio 

# Define a custom object for saving and loading the model
class CustomObjectScope:
    def __enter__(self):
        self.old_getattr = getattr
        setattr(tf.keras.layers, 'TransformerBlock', TransformerBlock)
        setattr(tf.keras.layers, 'TokenAndPositionEmbedding', TokenAndPositionEmbedding)
        setattr(tf.keras, 'NERModel', NERModel)
        setattr(tf.keras.losses, 'CustomNonPaddingTokenLoss', CustomNonPaddingTokenLoss)

    def __exit__(self, type, value, traceback):
        setattr(tf.keras.layers, 'TransformerBlock', self.old_getattr(tf.keras.layers, 'TransformerBlock'))
        setattr(tf.keras.layers, 'TokenAndPositionEmbedding', self.old_getattr(tf.keras.layers, 'TokenAndPositionEmbedding'))
        setattr(tf.keras, 'NERModel', self.old_getattr(tf.keras, 'NERModel'))
        setattr(tf.keras.losses, 'CustomNonPaddingTokenLoss', self.old_getattr(tf.keras.losses, 'CustomNonPaddingTokenLoss'))