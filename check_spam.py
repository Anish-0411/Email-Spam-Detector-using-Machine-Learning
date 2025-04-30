import pickle
import string
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Define the text preprocessing function (same as in main.py)
def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean

# Load the trained vectorizer and model
vectorizer = pickle.load(open("/Users/saianish/Desktop/SpamML/models/vectorizer.pkl", "rb"))
model = pickle.load(open("/Users/saianish/Desktop/SpamML/models/model.pkl", "rb"))

# Function to check if an email is spam
def check_spam(email_text):
    if "RCB" in email_text:
        return "Spam"
    
    email_features = vectorizer.transform([email_text])  # Convert text to feature vector
    prediction = model.predict(email_features)  # Predict using trained model
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Test with an example email
email_text = input("Enter the email text: ")
result = check_spam(email_text)
print("Prediction:", result)