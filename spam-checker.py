import tkinter as tk
from tkinter import messagebox
import pickle
import string
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
nltk.download('stopwords')

def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean

# Load the trained vectorizer and model
vectorizer = pickle.load(open("/Users/saianish/Desktop/SpamML/models/vectorizer.pkl", "rb"))
model = pickle.load(open("/Users/saianish/Desktop/SpamML/models/model.pkl", "rb"))

def check_spam():
    email_text = text_box.get("1.0", "end").strip()  # Get text from text box
    if not email_text:
        messagebox.showerror("Error", "Please enter some text!")
        return
    else:
        email_features = vectorizer.transform([email_text])  # Convert text to feature vector
        prediction = model.predict(email_features)  # Predict using trained model
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        messagebox.showinfo("Prediction", f"The email is: {result}")

# Create UI
root = tk.Tk()
root.title("Spam Checker")
root.geometry("400x300")

tk.Label(root, text="Enter Email Text:", font=("Arial", 12)).pack(pady=10)
text_box = tk.Text(root, height=8, width=40)
text_box.pack(pady=5)

check_button = tk.Button(root, text="Check Spam", command=check_spam, font=("Arial", 12))
check_button.pack(pady=10)

root.mainloop()