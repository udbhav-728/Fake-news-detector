import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import nltk
from nltk.corpus import stopwords
import string

# Set page title and icon
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Download NLTK stopwords (run once)
nltk.download('stopwords')

# --- Load or Mock Dataset ---
@st.cache_data
def load_data():
    # Replace with your actual dataset path (CSV with 'headline' and 'label' columns)
    try:
        data = pd.read_csv(r"C:\Users\C LAB-6\Downloads\blackbox-output-code-JMEMSVY4CV.csv")  # Replace with your file
    except FileNotFoundError:
        # Mock data for demo (if no file exists)
        st.warning("Using mock data. Replace with your CSV file.")
        data = pd.DataFrame({
            "headline": [
                "Scientists confirm climate change is accelerating",
                "Aliens land in New York, says government insider",
                "New vaccine proves 100% effective in trials",
                "Celebrity denies fake scandal created by AI"
            ],
            "label": [1, 0, 1, 0]  # 1=Real, 0=Fake
        })
    return data

# --- Preprocess Text ---
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# --- Train Model ---
@st.cache_data
def train_model(data):
    # Preprocess data
    data['cleaned_headline'] = data['headline'].apply(preprocess_text)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data['cleaned_headline'])
    y = data['label']
    
    # Train SVM
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    
    return model, vectorizer

# --- Streamlit UI ---
st.title("📰 Fake News Detector")
st.markdown("Enter a news headline to check if it's likely **real** or **fake**.")

# Load data and train model
data = load_data()
model, vectorizer = train_model(data)

# User input
user_input = st.text_area("Paste a news headline here:", "e.g., 'World leaders agree on new climate pact'")

# Predict button
if st.button("Check Headline"):
    if user_input.strip() == "":
        st.error("Please enter a headline!")
    else:
        # Preprocess and predict
        processed_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([processed_input])
        prediction = model.predict(vectorized_input)[0]
        proba = model.predict_proba(vectorized_input)[0]
        
        # Display result
        st.subheader("Result")
        if prediction == 1:
            st.success(f"✅ **Real** (Confidence: {proba[1]:.2%})")
        else:
            st.error(f"❌ **Fake** (Confidence: {proba[0]:.2%})")

# --- Optional: Show dataset sample ---
with st.expander("🔍 See sample training data"):
    st.dataframe(data.head())
