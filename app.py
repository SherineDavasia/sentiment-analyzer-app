import streamlit as st
import joblib 

# Load the trained model
model = joblib.load('sentiment_model.pkl')

# App title
st.title("✈️ Airline Tweet Sentiment Analyzer")
st.subheader("Tell me the vibe of your tweet 👇")

# User input
user_input = st.text_area("Tweet here......:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Type someting to tweet dear 😅")
    else:
        prediction = model.predict([user_input])[0]
        
        if prediction == 'positive':
            st.success("Tweet is Positive 💚")
        elif prediction == 'neutral':
            st.info("Tweet is Neutral 🤔")
        else:
            st.error("Tweet is Negative ❤️‍🩹")