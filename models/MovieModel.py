import tensorflow as tf
import tensorflow_recommenders as tfrs

class MovieModel(tf.keras.Model):
    def __init__(self, unique_movie_titles, embedding_dim=32):
        super().__init__()
        self.movie_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dim)
        ])

    def call(self, inputs):
        return self.movie_embedding(inputs)