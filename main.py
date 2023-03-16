import streamlit as st
from hollywood import hollywood_prediction
from bollywood import bollywood_prediction

st.set_page_config(
    page_title="Recommender System",
    page_icon=":sunglasses:",
    layout="wide",
    initial_sidebar_state="expanded",
   
)


def main_page():
    st.title("Recommendation system :sunglasses:")
    st.write("This is a movie recommender system for predicting both Hollywood movies and Bollywood movies")
    st.write("You have to write or select any movie and the model will suggest you five movies which you may like because you liked the selected movie.")
page_names_to_funcs = {
    "✅ Recommender System": main_page,
    "🔍 Bollywood Movie Prediction":bollywood_prediction,
    "🔍 Hollywood Movie Prediction":hollywood_prediction,
    
}
selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()