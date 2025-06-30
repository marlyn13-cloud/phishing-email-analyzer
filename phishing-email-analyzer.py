import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def clean_text(text):
    """
    Cleans the input text by removing punctuation, converting to lowercase,
    and removing stopwords.
    
    Args:
        text (str): The input text to clean.
        
    Returns:
        str: The cleaned text.
    """
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(cleaned_tokens)

#Extracts cleaned text from a DataFrame column
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
df=pd.read_csv('/kaggle/input/phishingemails/Phishing_Email.csv')
df.describe()

# Clean the text column
df["cleaned_text"] = df["text"].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["cleaned_text"])

#Model Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report as Classification_report
X_train, X_test, y_train, y_test = train_test_split(X, df["label"], test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

#predictions
y_pred = clf.predict(X_test)
print(Classification_report(y_test, y_pred))

def predict_email(text):
    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean])
    prediction = clf.predict(vec)
    return "Phishing" if prediction == 1 else "Legitimate"
