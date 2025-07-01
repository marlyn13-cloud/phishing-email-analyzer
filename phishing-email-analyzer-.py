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
        "Join our community and start networking today!",
        "Click here to verify your identity of your account.",
        "Urgent: Your payment information needs to be updated immediately.",
        "Congratulations! You've been selected for a special offer. Claim your reward now.",
        "Your meeting is confirmed for next Tuesday at 3 PM.",
        "Thank you for your recent purchase. Your order will be shipped soon.",
        "Join us for our upcoming webinar on cybersecurity trends. Register now to secure your spot."
    ],
    "label": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0]  # 1 = Phishing, 0 = Legitimate
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

import tkinter as tk

# ---------- GUI Code ---------- #
def analyze():
    email = email_input.get("1.0", tk.END).strip()
    if not email:
        result_label.config(text="Please enter some email text.")
    else:
        prediction = predict_email(email)
        result_label.config(text=f"Prediction: {prediction}")

# GUI window
root = tk.Tk()
root.title("Phishing Email Analyzer")
root.geometry("600x400")
root.config(bg="#eebaba")

# Email input box
email_input = tk.Text(root, height=10, width=60, font=("Arial", 12), wrap="word")
email_input.pack(pady=20)

# Analyze button
analyze_button = tk.Button(root, text="Analyze", command=analyze,
                           bg="#001aff", fg="white", font=("Arial", 12), padx=10, pady=5)
analyze_button.pack()

# Result label
result_label = tk.Label(root, text="", font=("Arial", 14), bg="#f4f4f4", fg="blue")
result_label.pack(pady=20)

root.mainloop()

#executes program for other users
from cx_Freeze import setup, Executable     
setup(
    name = "phishing email detector",
    version = "0.1",
    description = "This program detects phishing emails using machine learning.",
    executables = [Executable("phishing-email-analyzer.py")]
)
