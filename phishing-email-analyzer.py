import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
nltk.download('punkt')
nltk.download('stopwords')

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

# Example data
data = {
    "text": [
        "Dear user, your account has been compromised. Click here to secure it.",
        "Meeting scheduled for tomorrow at 10 AM.",
        "Congratulations! You've won a prize. Claim it now.",
        "Please review the attached document for your records.",
        "Urgent: Your payment is overdue. Update your information immediately.",
        "Join us for a webinar on data science next week.",
        "Your subscription has been renewed successfully.",
        "Verify your identity to continue using our services.",
        "Limited time offer: Get 50 percent off your next purchase!",
        "Your package has been shipped and is on its way.",
        "Security alert: Unusual login detected. Check your account.",
        "Reminder: Your appointment is scheduled for next Monday.",
        "Exclusive deal just for you! Click to learn more.",
        "Your account statement is ready for viewing.",
        "Action required: Update your billing information.",
        "Invitation to connect on LinkedIn.",
        "Your password will expire soon. Change it now.",
        "Thank you for your purchase! Your order is confirmed.",
        "Alert: Suspicious activity detected in your account.",
        "Join our community and start networking today!"
    ],
    "label": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Phishing, 0 = Legitimate
}

df = pd.DataFrame(data)

# Clean the text column
df["cleaned_text"] = df["text"].apply(clean_text)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["cleaned_text"])

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, df["label"], test_size=0.2, stratify=df["label"], random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=1))

def predict_email(text):
    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean])
    prediction = clf.predict(vec)
    return "Phishing" if prediction == 1 else "Legitimate"

# Example usage
print(predict_email("Please verify your account information"))

#UI
import tkinter as tk     
def predict():
    email_text = text_entry.get("1.0", tk.END)
    prediction = predict_email(email_text)
    result_label.config(text=f"Prediction: {prediction}")

root = tk.Tk()
root.title("Phishing Email Predictor")

text_entry = tk.Text(root, height=10, width=50, bg="lightgrey", fg="black", font=("Arial", 12))
text_entry.pack()

predict_button = tk.Button(root, text="Predict", command=predict, bg="blue", fg="white", font=("Arial", 12))
predict_button.pack()

result_label = tk.Label(root, text="", bg ="white", fg="red", font=("Arial", 12))
result_label.pack()

root.mainloop()
