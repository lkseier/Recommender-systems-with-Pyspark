import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from src.recommender_utils import cast_columns, drop_missing, get_data_path

# Helper to get movie titles (if available)
def get_movie_titles(spark, movie_ids):
    try:
        movies_path = get_data_path('movies.csv')
        movies_df = spark.read.csv(movies_path, header=True, inferSchema=True)
        return dict(movies_df.rdd.map(lambda row: (row['movieId'], row['title'])).collect())
    except Exception:
        return {mid: f"Movie {mid}" for mid in movie_ids}

def recommend_for_user(model, spark, user_id, num_recs=5):
    # Create DataFrame for the user
    user_df = spark.createDataFrame([(user_id,)], ['userId'])
    recs = model.recommendForUserSubset(user_df, num_recs).collect()
    if not recs:
        print(f"No recommendations found for user {user_id}.")
        return
    recs = recs[0]['recommendations']
    movie_ids = [r['movieId'] for r in recs]
    ratings = [r['rating'] for r in recs]
    titles = get_movie_titles(spark, movie_ids)
    print(f"Top {num_recs} recommendations for user {user_id}:")
    for mid, score in zip(movie_ids, ratings):
        print(f"{titles.get(mid, mid)} (Predicted rating: {score:.2f})")

def main():
    if len(sys.argv) < 2:
        print("Usage: python recommend.py <user_id>")
        sys.exit(1)
    user_id = int(sys.argv[1])
    spark = SparkSession.builder.appName('RecommenderRecommend').getOrCreate()
    # Load model
    model = ALSModel.load('als_model')
    recommend_for_user(model, spark, user_id)
    spark.stop()

if __name__ == '__main__':
    main()
