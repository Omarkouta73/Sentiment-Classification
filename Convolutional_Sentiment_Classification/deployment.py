import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import gensim
import pickle
import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors


def sentence_to_vec(sentence, model, vector_size=300):
    words = sentence.lower().split()
    word_vecs = []

    for word in words:
        if word in model:
            word_vecs.append(model[word])
    
    if len(word_vecs) == 0:
        return np.zeros(vector_size)  
    
    word_vecs = np.array(word_vecs)
    return np.mean(word_vecs, axis=0)


def predict_sentiment(sentence, model, word2vec_model, label_encoder):

    sentence_vec = np.array([sentence_to_vec(sentence, word2vec_model)])
    
    # Make prediction
    prediction = model.predict(sentence_vec)[0]
    predicted_class = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    confidence = np.max(prediction) * 100
    
    return {
        'predicted_label': predicted_label,
        'confidence': confidence,
        'probabilities': {label: float(prediction[i]) for i, label in enumerate(label_encoder.classes_)}
    }

# Streamlit app
def main():
    st.set_page_config(
        page_title="Sentiment Analysis App",
        page_icon="😊",
        layout="centered"
    )
    
    st.title("Sentiment Analysis with 1D CNN")
    st.write("Enter a sentence to analyze its sentiment")

    @st.cache_resource
    def load_dependencies():
        try:
            # Load the trained CNN model
            model = load_model('sentiment_cnn_model3_gpu.h5')
            
            word2vec_model = KeyedVectors.load("fasttext-wiki-news-subwords-300.model")
            
            with open('label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
                
            return model, word2vec_model, label_encoder
            
        except Exception as e:
            st.error(f"Error loading model dependencies: {e}")
            return None, None, None
    
    model, word2vec_model, label_encoder = load_dependencies()
    
    if model is None or word2vec_model is None or label_encoder is None:
        st.error("Please make sure you've saved the model files before running this app")
        return
    
    user_input = st.text_area("Enter text:", height=100, 
                             placeholder="Type your text here... (Example: I like this movie very much!)")
    
    if st.button("Analyze Sentiment"):
        if not user_input:
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing sentiment..."):
                # Get prediction
                if getattr(st.session_state, 'demo_mode', False):
                    # Demo mode: Random predictions
                    import random
                    result = {
                        'predicted_label': random.choice(['positive', 'neutral', 'negative']),
                        'confidence': random.uniform(70, 99),
                        'probabilities': {
                            'positive': random.random(),
                            'neutral': random.random(),
                            'negative': random.random()
                        }
                    }

                    sum_probs = sum(result['probabilities'].values())
                    for k in result['probabilities']:
                        result['probabilities'][k] /= sum_probs
                else:
    
                    result = predict_sentiment(user_input, model, word2vec_model, label_encoder)
                
    
                sentiment = result['predicted_label']
                confidence = result['confidence']
                
                # Display sentiment with appropriate emoji and color
                if sentiment == 'positive':
                    emoji = "😊"
                    color = "green"
                elif sentiment == 'neutral':
                    emoji = "😐"
                    color = "gray"
                else:  # negative
                    emoji = "😟"
                    color = "red"
                
                st.markdown(f"### Sentiment: <span style='color:{color}'>{sentiment} {emoji}</span>", unsafe_allow_html=True)
                st.progress(confidence/100)
                st.write(f"Confidence: {confidence:.2f}%")

                st.write("Probability breakdown:")
                for label, prob in result['probabilities'].items():
                    st.write(f"- {label}: {prob*100:.2f}%")
    
    # Example sentences section
    with st.expander("Try example sentences"):
        examples = [
            "I love this movie! It's absolutely amazing.",
            "This film was just okay, nothing special.",
            "Terrible movie, complete waste of time.",
            "The acting was good but the plot was confusing.",
        ]
        for example in examples:
            if st.button(example):
                st.session_state.example_text = example
                st.experimental_rerun()
    

    if 'example_text' in st.session_state:
        st.text_area("Enter text:", height=100, value=st.session_state.example_text, key="user_input_example")

        st.session_state.pop('example_text')
    
    # Footer
    st.markdown("---")
    st.markdown("Sentiment Analysis App | Created with Streamlit and TensorFlow")

if __name__ == "__main__":
    main()