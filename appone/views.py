from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from appone.forms import MovieRecommendForm
from django.views.generic import View, TemplateView

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

#########################################################

#########################################################

# Create your views here.

class HomePage(TemplateView):
    template_name = 'appone/index.html'

class MoviePlot(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'appone/movieplot.html')
    
    def post(self, request, *args, **kwargs):
        form = MovieRecommendForm(request.POST)
        input = request.POST.get('input')

        def func(inp):
            credits_df = pd.read_csv("static/css/tmdb_5000_credits.csv")
            movies_df = pd.read_csv("static/css/tmdb_5000_movies.csv")

            movies_df.head()
            movies_df.head()

            credits_df.columns = ['id','tittle','cast','crew']
            movies_df = movies_df.merge(credits_df, on="id")

            movies_df.head()

            C = movies_df["vote_average"].mean()
            m = movies_df["vote_count"].quantile(0.9)

            new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]

            def weighted_rating(x, C=C, m=m):
                v = x["vote_count"]
                R = x["vote_average"]

                return (v/(v + m) * R) + (m/(v + m) * C)

            new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
            new_movies_df = new_movies_df.sort_values('score', ascending=False)

            new_movies_df[["title", "vote_count", "vote_average", "score"]].head(10)

            def plot():
                popularity = movies_df.sort_values("popularity", ascending=False)
                plt.figure(figsize=(12, 6))
                plt.barh(popularity["title"].head(10), popularity["popularity"].head(10), align="center", color="skyblue")
                plt.gca().invert_yaxis()
                plt.title("Top 10 movies")
                plt.xlabel("Popularity")
                plt.show()
                
            tfidf = TfidfVectorizer(stop_words="english")
            movies_df["overview"] = movies_df["overview"].fillna("")

            tfidf_matrix = tfidf.fit_transform(movies_df["overview"])

            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            # print(cosine_sim.shape)

            indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()
            def get_recommendations(title, cosine_sim=cosine_sim):
                """
                in this function,
                    we take the cosine score of given movie
                    sort them based on cosine score (movie_id, cosine_score)
                    take the next 10 values because the first entry is itself
                    get those movie indices
                    map those indices to titles
                    return title list
                """
                idx = indices[title]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:11]
                # (a, b) where a is id of movie, b is sim_score
                movies_indices = [ind[0] for ind in sim_scores]
                movies = movies_df["title"].iloc[movies_indices]
                return movies.values[0:10]
            response = get_recommendations(inp)
            return response

        countfull = [1,2,3,4,5,6,7,8,9,10]
        response = func(input)
        data=zip(countfull,response)
        return render(request, 'appone/movieplot.html', {'data':data, 'inp':input, 
        'check':response[0]})


class MovieKey(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'appone/moviekey.html')
    
    def post(self, request, *args, **kwargs):
        form = MovieRecommendForm(request.POST)
        input = request.POST.get('input')

        def func(inp):
            credits_df = pd.read_csv("static/css/tmdb_5000_credits.csv")
            movies_df = pd.read_csv("static/css/tmdb_5000_movies.csv")

            movies_df.head()
            movies_df.head()

            credits_df.columns = ['id','tittle','cast','crew']
            movies_df = movies_df.merge(credits_df, on="id")

            movies_df.head()

            C = movies_df["vote_average"].mean()
            m = movies_df["vote_count"].quantile(0.9)

            new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]

            def weighted_rating(x, C=C, m=m):
                v = x["vote_count"]
                R = x["vote_average"]

                return (v/(v + m) * R) + (m/(v + m) * C)

            new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
            new_movies_df = new_movies_df.sort_values('score', ascending=False)

            new_movies_df[["title", "vote_count", "vote_average", "score"]].head(10)

            def plot():
                popularity = movies_df.sort_values("popularity", ascending=False)
                plt.figure(figsize=(12, 6))
                plt.barh(popularity["title"].head(10), popularity["popularity"].head(10), align="center", color="skyblue")
                plt.gca().invert_yaxis()
                plt.title("Top 10 movies")
                plt.xlabel("Popularity")
                plt.show()
                
            tfidf = TfidfVectorizer(stop_words="english")
            movies_df["overview"] = movies_df["overview"].fillna("")

            tfidf_matrix = tfidf.fit_transform(movies_df["overview"])

            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            # print(cosine_sim.shape)

            indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()
            def get_recommendations(title, cosine_sim=cosine_sim):
                """
                in this function,
                    we take the cosine score of given movie
                    sort them based on cosine score (movie_id, cosine_score)
                    take the next 10 values because the first entry is itself
                    get those movie indices
                    map those indices to titles
                    return title list
                """
                idx = indices[title]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:11]
                # (a, b) where a is id of movie, b is sim_score
                movies_indices = [ind[0] for ind in sim_scores]
                movies = movies_df["title"].iloc[movies_indices]
                return movies.values[0:10]
            
            features = ["cast", "crew", "keywords", "genres"]

            for feature in features:
                movies_df[feature] = movies_df[feature].apply(literal_eval)

            movies_df[features].head(10)
            def get_director(x):
                for i in x:
                    if i["job"] == "Director":
                        return i["name"]
                return np.nan

            def get_list(x):
                if isinstance(x, list):
                    names = [i["name"] for i in x]

                    if len(names) > 3:
                        names = names[:3]

                    return names

                return []

            movies_df["director"] = movies_df["crew"].apply(get_director)

            features = ["cast", "keywords", "genres"]
            for feature in features:
                movies_df[feature] = movies_df[feature].apply(get_list)

            movies_df[['title', 'cast', 'director', 'keywords', 'genres']].head()

            def clean_data(x):
                if isinstance(x, list):
                    return [str.lower(i.replace(" ", "")) for i in x]
                else:
                    if isinstance(x, str):
                        return str.lower(x.replace(" ", ""))
                    else:
                        return ""

            features = ['cast', 'keywords', 'director', 'genres']
            for feature in features:
                movies_df[feature] = movies_df[feature].apply(clean_data)


            def create_soup(x):
                return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


            movies_df["soup"] = movies_df.apply(create_soup, axis=1)

            count_vectorizer = CountVectorizer(stop_words="english")
            count_matrix = count_vectorizer.fit_transform(movies_df["soup"])

            # print(count_matrix.shape)

            cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
            # print(cosine_sim2.shape)

            movies_df = movies_df.reset_index()
            indices = pd.Series(movies_df.index, index=movies_df['title'])

            response = get_recommendations(inp, cosine_sim2)
            return response

        countfull = [1,2,3,4,5,6,7,8,9,10]
        response = func(input)
        data=zip(countfull,response)
        return render(request, 'appone/moviekey.html', {'data':data, 'inp':input, 
        'check':response[0]})

class TopTenVoted(View):
    def get(self, request, *args, **kwargs):
        def func():
            credits_df = pd.read_csv("static/css/tmdb_5000_credits.csv")
            movies_df = pd.read_csv("static/css/tmdb_5000_movies.csv")

            movies_df.head()
            movies_df.head()

            credits_df.columns = ['id','tittle','cast','crew']
            movies_df = movies_df.merge(credits_df, on="id")

            movies_df.head()

            C = movies_df["vote_average"].mean()
            m = movies_df["vote_count"].quantile(0.9)

            new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]

            def weighted_rating(x, C=C, m=m):
                v = x["vote_count"]
                R = x["vote_average"]

                return (v/(v + m) * R) + (m/(v + m) * C)

            new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
            new_movies_df = new_movies_df.sort_values('score', ascending=False)
            l1 = new_movies_df["title"].head(10).values[0:10]
            l2 = new_movies_df["vote_count"].head(10).values[0:10]
            l3 = new_movies_df["vote_average"].head(10).values[0:10]
            l4 = new_movies_df["score"].head(10).values[0:10]
            l5 = [1,2,3,4,5,6,7,8,9,10]
            data = zip(l1,l2,l3,l4,l5)
            return data
        
        response = func()
        return render(request, 'appone/toptenvoted.html', {'data':response})

class TopTenPopular(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'appone/toptenpopular.html')
