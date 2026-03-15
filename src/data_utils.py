import numpy as np
import tensorflow_datasets as tfds

def load_and_prep_data():
    ratings = tfds.load("movielens/100k-ratings", split="train",shuffle_files=True)
    movies = tfds.load("movielens/100k-movies", split="train",shuffle_files=True)

    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
    })
    movies = movies.map(lambda x: x["movie_title"])

    unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(lambda x: x["user_id"]))))
    unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1_000))))

    return ratings, movies, unique_user_ids, unique_movie_titles