import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Text Preprocessing Function (same as before) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def upgraded_preprocess_text(text):
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.replace('\r\n', ' ')
    text = re.sub(r'http\S+|@\S*\s?|\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(cleaned_tokens)

# --- Load the saved model and vectorizer ---
with open('classifier_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

# --- Create the Prediction Function ---
def predict_category(resume_text):
    # 1. Clean the input text
    cleaned_text = upgraded_preprocess_text(resume_text)
    
    # 2. Transform the text using the loaded vectorizer
    # Note: We use transform(), not fit_transform(), as we're using the existing vocabulary.
    # The input needs to be in a list or iterable.
    vectorized_text = loaded_vectorizer.transform([cleaned_text])
    
    # 3. Make a prediction using the loaded model
    prediction = loaded_model.predict(vectorized_text)
    
    return prediction[0] # The result is an array, so we take the first element.

# --- Example Usage ---
if __name__ == '__main__':
    sample_resume = """
    John Doe | Data Analyst | (123) 456-7890 | john.doe@email.com
    
    Experience with Python, Pandas, and SQL to analyze complex datasets.
    Created dashboards in Tableau to visualize key performance indicators.
    Expert in statistical analysis and data modeling.
    """
    
    predicted_role = predict_category(sample_resume)
    print(f"The predicted job category for the resume is: {predicted_role}")