import streamlit as st
import torch
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification
import sys
import os

# Add the current directory to the path so we can import the Model class
sys.path.append(os.path.dirname(__file__))

# Import the model class
from Model import BertForSentimentClassification

# Set page title and configuration
st.set_page_config(
    page_title="Arabic Sentiment Analysis",
    page_icon="🔍",
    layout="wide"
)

# Function to load the model
@st.cache_resource
def load_model():
    model_name = "asafaya/bert-base-arabic"
    model_path = "results/Arabic-Sentiment-CLF/ar_bert_model.bin"
    
    # Create the model
    config = AutoConfig.from_pretrained("asafaya/bert-base-arabic", num_labels=3)  # or whatever base model you used
    model = AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(torch.load(model_path), strict=False)
    print(model)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

# Function to predict sentiment
def predict_sentiment(text, model, tokenizer):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
    
    # Map prediction to sentiment
    sentiment_map = {0: "Positive", 1: "Neutral", 2: "Negative"}
    return sentiment_map[prediction], confidence

# Main function
def main():
    # Page title and description
    st.title("🔍 Arabic Sentiment Analysis")
    st.markdown("""
    This app analyzes the sentiment of Arabic text using a fine-tuned BERT model.
    Enter your text below to see if it's positive, neutral, or negative.
    """)
    
    # Load model
    with st.spinner("Loading model... (this may take a moment)"):
        try:
            model, tokenizer = load_model()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("Make sure the model files are in the correct location: './Arabic-Sentiment-CLF/ar_bert_model.bin'")
            return
    
    # Text input
    text_input = st.text_area("Enter Arabic text:", height=150, 
                             placeholder="أدخل النص العربي هنا...")
    
    # Examples
    st.sidebar.header("Examples")
    examples = [
        "أنا أحب هذا المكان كثيرا",
        "الطقس معتدل اليوم",
        "هذا المنتج سيء للغاية، لا أنصح به أبدا"
    ]
    
    for example in examples:
        if st.sidebar.button(example):
            st.session_state.text_input = example
            text_input = example
    
    # Analyze button
    if st.button("Analyze Sentiment", type="primary") or text_input:
        if text_input:
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence = predict_sentiment(text_input, model, tokenizer)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display sentiment with appropriate emoji and color
                    if sentiment == "Positive":
                        st.markdown(f"### Sentiment: <span style='color:green'>**{sentiment}** 😊</span>", unsafe_allow_html=True)
                    elif sentiment == "Neutral":
                        st.markdown(f"### Sentiment: <span style='color:blue'>**{sentiment}** 😐</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"### Sentiment: <span style='color:red'>**{sentiment}** 😠</span>", unsafe_allow_html=True)
                
                with col2:
                    # Display confidence percentage
                    confidence_pct = confidence * 100
                    st.markdown(f"### Confidence: **{confidence_pct:.2f}%**")
                
                # Create a progress bar for confidence
                st.progress(confidence)
        else:
            st.warning("Please enter some text to analyze.")
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info("""
    This app uses a fine-tuned BERT model for Arabic sentiment analysis.
    The model was trained on the ArSAS dataset, which contains Arabic tweets
    labeled with sentiment (Positive, Neutral, Negative).
    """)

if __name__ == "__main__":
    main()