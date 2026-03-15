import tensorflow as tf
import tensorflow_recommenders as tfrs
from src.data_utils import load_and_prep_data
from models.UserModel import UserModel
from models.MovieModel import MovieModel
from models.MovielensModel import MovielensModel
import os
import logging
logger = logging.getLogger(__name__)

def train_eval():
    ratings, movies, unique_user_ids, unique_movie_titles = load_and_prep_data()

    embedding_dim = 32
    user_model = UserModel(unique_user_ids, embedding_dim)
    movie_model = MovieModel(unique_movie_titles, embedding_dim)

    task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(movie_model)
        )
    )

    model = MovielensModel(user_model, movie_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    class LogCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logger.info(f"Epoch {epoch+1}: Loss = {logs['loss']:.4f}")

    cached_train = ratings.shuffle(100_000).batch(8192).cache()
    model.fit(cached_train, epochs=3 , callbacks=[LogCallback()])

    print("Training complete")
    cached_test = ratings.batch(4096).cache()

    metrics = model.evaluate(cached_test, return_dict=True)

    print(f"\n--- ผลการวัดผล (Evaluation) ---")
    print(f"Top-100 Accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.4f}")
    print(f"Top-10 Accuracy: {metrics['factorized_top_k/top_10_categorical_accuracy']:.4f}")

    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

    index.index_from_dataset(
    tf.data.Dataset.zip((
        movies.batch(100), 
        movies.batch(100).map(model.movie_model)
    ))
    )

    user_id_to_predict = "42"
    scores, titles = index(tf.constant([user_id_to_predict]))

    print(f"\n--- คำแนะนำสำหรับ User {user_id_to_predict} ---")
    for i, title in enumerate(titles[0, :5]):
        print(f"{i+1}: {title.numpy().decode('utf-8')}")

    path = os.path.join("saved_model", "my_recommender")
    tf.saved_model.save(index, path)
    print(f"Model path: {path}")