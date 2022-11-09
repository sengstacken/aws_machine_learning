import tensorflow as tf
import numpy as np
import os
import argparse
from typing import Dict, Text
import subprocess
import sys
import tempfile

print(tf.__version__)

# Workaround for dependencies
def install_pip(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
install_pip('tensorflow_recommenders')
import tensorflow_recommenders as tfrs

install_pip('tensorflow-datasets')
import tensorflow_datasets as tfds

install_pip('scann')
from scann import scann_ops

print(tfrs.__version__)
print(tfds.__version__)

# keras model, two-tower
class MovielensModel(tfrs.Model):
    def __init__(self, unique_movie_titles,unique_user_ids,movies):
        super().__init__()
        embedding_dimension = 32
        # Set up a model for representing movies.
        self.movie_model = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
          # We add an additional embedding to account for unknown tokens.
          tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])

        # Set up a model for representing users.
        self.user_model = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
            # We add an additional embedding to account for unknown tokens.
          tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        # Set up a task to optimize the model and compute metrics.
        self.task = tfrs.tasks.Retrieval(
          metrics=tfrs.metrics.FactorizedTopK(
            candidates=(
                movies
                .batch(128)
                .cache()
                .map(lambda title: (title, self.movie_model(title)))
            )
          )
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["movie_title"])

        # The task computes the loss and the metrics.

        return self.task(
            user_embeddings,
            positive_movie_embeddings,
            candidate_ids=features["movie_title"],
            compute_metrics=not training
        )

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8192, metavar='N',
                        help='input batch size for training (default: 8192)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=54321, metavar='S',
                        help='random seed (default: 54321)')
    parser.add_argument('--bf', type=bool, default=True)
    
    # Container environment
    #parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
#     parser.add_argument('--ratings-data', type=str, default=os.environ['SM_CHANNEL_RATINGS'])
#     parser.add_argument('--movies-data', type=str, default=os.environ['SM_CHANNEL_MOVIES'])
    
    args = parser.parse_args()

    # data
    # Load the MovieLens 100K data.
    ratings = tfds.load(
        "movielens/100k-ratings",
        split="train"
    )

    # Get the ratings data.
    ratings = (ratings
               # Retain only the fields we need.
               .map(lambda x: {"user_id": x["user_id"], "movie_title": x["movie_title"]})
               # Cache for efficiency.
               .cache(tempfile.NamedTemporaryFile().name)
    )

    # Get the movies data.
    movies = tfds.load("movielens/100k-movies", split="train")
    movies = (movies
              # Retain only the fields we need.
              .map(lambda x: x["movie_title"])
              # Cache for efficiency.
              .cache(tempfile.NamedTemporaryFile().name))
    
#     ratings = tf.data.Dataset.load(args.ratings_data)
#     movies = tf.data.Dataset.load(args.movies_data)

    # set up user and movie vocabularies
    user_ids = ratings.map(lambda x: x["user_id"])
    unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
    unique_user_ids = np.unique(np.concatenate(list(user_ids.batch(1000))))

    # set up train / test sets
    tf.random.set_seed(args.seed)
    shuffled = ratings.shuffle(100_000, seed=args.seed, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)

    # model
    model = MovielensModel(unique_movie_titles,unique_user_ids,movies)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=args.lr))
    model.fit(train.batch(args.batch_size), epochs=args.epochs, verbose=2)
    print('Evaluating the model')
    eval = model.evaluate(test.batch(args.batch_size), return_dict=True)
    print(eval)
    
    if args.bf is True:

        # build the brute_force look up
        brute_force = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        brute_force.index_from_dataset(
                movies.batch(128).map(lambda title: (title, model.movie_model(title)))
        )

        print('Testing the model')
        # Get predictions for user 42.
        _, titles = brute_force(np.array(["42"]))
        print(f"Top recommendations: {titles[0]}")

        print('Saving the model')
        tf.saved_model.save(brute_force,os.path.join(args.sm_model_dir, "1"))

    else:
    
        # build the scann look up
        # We re-index the ScaNN layer to include the user embeddings in the same model.
        # This way we can give the saved model raw features and get valid predictions
        # back.
        scanner = tfrs.layers.factorized_top_k.ScaNN(model.user_model, num_reordering_candidates=500, num_leaves_to_search=30)
        scanner.index_from_dataset(movies.batch(128).map(lambda title: (title, model.movie_model(title))))

        print('Testing the model')
        # Get predictions for user 42.
        _, titles = scanner(np.array(["42"]))
        print(f"Top recommendations: {titles[0]}")

        print('Saving the model')
        tf.saved_model.save(
            scanner,
            os.path.join(args.sm_model_dir, "1"),
            options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
        )

if __name__ == '__main__':
    main()