import pickle

# Load the vectorizer
with open('/Users/saianish/Desktop/SpamML/models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the model
with open('/Users/saianish/Desktop/SpamML/models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example usage
email = ["Congratulations! You won a free ticket. Click here."]
X = vectorizer.transform(email)
prediction = model.predict(X)

print("Spam" if prediction[0] == 1 else "Not Spam")