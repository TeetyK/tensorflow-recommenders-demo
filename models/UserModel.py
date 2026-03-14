import tensorflow as tf

class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids,mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1 , embedding_dim)
        ])
    
    def call(self, inputs):
        return self.user_embedding(inputs)