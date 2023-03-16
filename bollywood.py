import numpy as np 
import streamlit as st

import pandas as pd
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def bollywood_prediction():
    st.title("Bollywood movies")
    movies=pd.read_csv('BollywoodMovieDetail.csv')
    movies=movies[['imdbId','title','genre','writers','actors','directors']]
    movies.isnull().sum()
    movies.dropna(inplace=True)

    movies['genre']=movies['genre'].apply(lambda x:x.replace(" ",""))
    movies['genre']=movies['genre'].apply(lambda x:x.split())
    movies['writers']=movies['writers'].apply(lambda x:x.replace(" ",""))
    movies['writers']=movies['writers'].apply(lambda x:x.split())
    movies['actors']=movies['actors'].apply(lambda x:x.replace(" ",""))
    movies['actors']=movies['actors'].apply(lambda x:x.split())
    movies['directors']=movies['directors'].apply(lambda x:x.replace(" ",""))
    movies['directors']=movies['directors'].apply(lambda x:x.split())
    movies['tags']=movies['genre']+movies['writers']+movies['actors']+movies['directors']
    new_df=new_df=movies[['imdbId','title','tags']]
    new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
    ps=PorterStemmer()
    def stem(text):
        y=[]
        
        for i in text.split():
            y.append(ps.stem(i))
            
        return " ".join(y)
    new_df['tags']=new_df['tags'].apply(stem)
    new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
    cv=CountVectorizer(max_features=5000,stop_words='english')
    vectors=cv.fit_transform(new_df['tags']).toarray()
    similarity=cosine_similarity(vectors)

    selected_movie_name = st.selectbox(
    'Type or select a movie from the dropdown',
    movies['title'].values)
    def recommend(selected_movie_name):
            a=[]
            movie_index=new_df[new_df['title']==selected_movie_name].index[0]
            distances=similarity[movie_index]
            movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
        
            for i in movie_list:
                a.append(new_df.iloc[i[0]].title)
            return a

    if st.button('Recommend'):
        x=recommend(selected_movie_name)
        for i in x:
            st.write(i)