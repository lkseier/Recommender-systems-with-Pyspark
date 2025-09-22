import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from src.recommender_utils import get_data_path

def get_movie_details(spark, movie_ids):
    try:
        # Load links.csv to map movieId -> tmdbId
        links_path = get_data_path('links.csv')
        links_df = spark.read.csv(links_path, header=True, inferSchema=True)
        links_df = links_df.filter(links_df['tmdbId'].rlike('^\\d+$'))
        links_df = links_df.withColumn('tmdbId', links_df['tmdbId'].cast('int'))
        movieid_to_tmdbid = dict(links_df.rdd.map(lambda row: (row['movieId'], row['tmdbId'])).collect())

        # Load movies_metadata.csv to map tmdbId -> details
        movies_path = get_data_path('movies_metadata.csv')
        movies_df = spark.read.csv(movies_path, header=True, inferSchema=True)
        movies_df = movies_df.filter(movies_df['id'].rlike('^\\d+$'))
        movies_df = movies_df.withColumn('id', movies_df['id'].cast('int'))
        # Build a dict: tmdbId -> (title, genres, poster_path)
        def parse_genres(genres_str):
            import ast
            try:
                genres = ast.literal_eval(genres_str) if genres_str else []
                if isinstance(genres, list):
                    return ', '.join([g['name'] for g in genres if 'name' in g])
            except Exception:
                pass
            return ''
        movie_map = {}
        for row in movies_df.collect():
            genres = parse_genres(row['genres']) if 'genres' in row else ''
            poster = row['poster_path'] if 'poster_path' in row else ''
            movie_map[row['id']] = (row['title'], genres, poster)

        # Compose mapping: movieId -> tmdbId -> (title, genres, poster)
        result = {}
        for mid in movie_ids:
            tmdbid = movieid_to_tmdbid.get(mid)
            details = movie_map.get(tmdbid) if tmdbid is not None else None
            if details:
                result[mid] = details
            else:
                result[mid] = (f"Movie {mid}", '', '')
        return result
    except Exception as e:
        print('Error in get_movie_details:', e)
        return {mid: (f"Movie {mid}", '', '') for mid in movie_ids}

def recommend_for_user(model, spark, user_id, num_recs=5):
    user_df = spark.createDataFrame([(user_id,)], ['userId'])
    recs = model.recommendForUserSubset(user_df, num_recs).collect()
    if not recs:
        return []
    recs = recs[0]['recommendations']
    movie_ids = [r['movieId'] for r in recs]
    ratings = [r['rating'] for r in recs]
    details = get_movie_details(spark, movie_ids)
    return [(details.get(mid, (mid, '', '')), score) for mid, score in zip(movie_ids, ratings)]

def main():
    st.title('Movie Recommender System (PySpark ALS)')
    user_id = st.number_input('Enter your user ID:', min_value=1, step=1)
    if st.button('Get Recommendations'):
        spark = SparkSession.builder.appName('RecommenderStreamlit').getOrCreate()
        model = ALSModel.load('als_model')
        recs = recommend_for_user(model, spark, int(user_id))
        if recs:
            st.write('Top Recommendations:')
            for (title, genres, poster), score in recs:
                st.markdown(f'**{title}** (Predicted rating: {score:.2f})')
                if genres:
                    st.write(f'Genres: {genres}')
                if poster:
                    st.image(f'https://image.tmdb.org/t/p/w200{poster}')
                st.write('---')
        else:
            st.write('No recommendations found for this user.')
        spark.stop()

if __name__ == '__main__':
    main()
