import streamlit as st
import numpy as np
import pandas as pd
import pickle
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def hollywood_prediction():
    st.title("Hollywood movies")
    
    movies=pd.read_csv('tmdb_5000_credits.csv')
    credits=pd.read_csv('tmdb_5000_movies.csv')

    ps=PorterStemmer()

    movies_dict=pickle.load(open('movies.pkl','rb'))
    new=pd.DataFrame(movies_dict)
    
    cv=CountVectorizer(max_features=5000,stop_words='english')
    vectors=cv.fit_transform(new['tags']).toarray()
    cv.get_feature_names_out()
    similarity=cosine_similarity(vectors)
    sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]
    
    def recommend(movies):
        
        a=[]
        movie_index=new[new['title']==movies].index[0]
        distances=similarity[movie_index]
        movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
        for i in movie_list:
            a.append(new.iloc[i[0]].title)
        return a

    movies = st.selectbox('Type or select a movie from the dropdown',new['title'].values)
    if st.button('Recommend'):
        x=recommend(movies)
        for i in x:
             st.write(i)
    

