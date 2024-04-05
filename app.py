# app.py
import streamlit as st
from flair.models import TextClassifier
from flair.data import Sentence

# Load the pre-trained Flair sentiment classifier
classifier = TextClassifier.load('en-sentiment')

def classify_review(review):
    sentence = Sentence(review)
    classifier.predict(sentence)
    result = sentence.labels[0]

    sentiment = result.value
    score = round(result.score * 5, 2)  # Scale the score to 0-5

    return sentiment, score

def main():
    st.title('Movie Review Sentiment Analysis')

    review = st.text_area('Enter your movie review here:')
    if st.button('Classify'):
        if not review:
            st.warning('Please enter a review.')
        else:
            sentiment, score = classify_review(review)
            st.write(f'**Sentiment:** {sentiment}')
            st.write(f'**Score:** {score}/5')

if __name__ == '__main__':
    main()
