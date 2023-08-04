import streamlit as st
from PIL import Image

st.title('Emotion Detection and Age Recognition')
st.subheader('Made by Purav and Shrivatsan')
st.divider()
st.subheader('About the project')
st.write("Our project uses two models to classify images in real time based on emotion and age using CNN's")
st.subheader('Model specifications')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Emotion detection')
    st.write(''' Emotion detection is done using a classification model that categorizes emotions into 7 categories: Happy, Sad, Angry, Disgust, Fear, Neutral ''')
    st.write("The model uses 2 sets of CNN's with 2 units each and a connected layer to achieve a final accuracy of 68.5 on the test set")
    st.image(Image.open("C:\\Users\\HP\\Desktop\\Screenshot 2023-08-04 214500.png"), caption="fard")

with col2:
    st.subheader('Age detection')
    st.write('Age detection is done using a regression model that predicts a persons ')

# st.image(Image.open("C:\\Users\\HP\\Desktop\\amongusre.png"), caption="fard")

st.write()  