# import ssl
# import nltk

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
#     ssl._create_default_https_context = _create_unverified_https_context
# except AttributeError:
#     pass

# nltk.download('stopwords', download_dir='/Users/saianish/nltk_data')
# nltk.data.path.append('/Users/saianish/nltk_data')


# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('stopwords')
# import nltk
# nltk.download('stopwords', download_dir='/Users/saianish/nltk_data')
# import nltk
# nltk.data.path.append('/Users/saianish/nltk_data')
# import nltk
# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('stopwords')
# nltk.data.path.append('/Users/saianish/nltk_data')


# import pickle
# import numpy as np
# import pandas as pd
# import seaborn as sns
# sns.set_style("white")
# import matplotlib.pyplot as plt
# import string
# from pickle import dump
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import nltk
# from nltk.corpus import stopwords
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# nltk.download('stopwords')



# # Load the dataset
# import os

# dataset_path = "/Users/saianish/Desktop/SpamML/dataset/emails.csv"

# if not os.path.exists(dataset_path):
#     print(f"Error: Dataset not found at {dataset_path}")
# else:
#     dataset = pd.read_csv(dataset_path)
# dataset = pd.read_csv('/Users/saianish/Desktop/SpamML/dataset/emails.csv')
# dataset.shape


# # Show dataset head (first 5 records)
# dataset.head() 

# # Show dataset info
# dataset.info()

# # Show dataset statistics
# dataset.describe()

# # Visualize spam  frequenices
# plt.figure(dpi=100)
# sns.countplot(dataset['spam'])
# plt.title("Spam Freqencies")
# plt.show()

# # Check for missing data for each column 
# dataset.isnull().sum()


# # Check for duplicates and remove them 
# dataset.drop_duplicates(inplace=True)


# # Cleaning data from punctuation and stopwords and then tokenizing it into words (tokens)
# def process(text):
#     nopunc = [char for char in text if char not in string.punctuation]
#     nopunc = ''.join(nopunc)
#     clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
#     return clean

# # Fit the CountVectorizer to data
# message = CountVectorizer(analyzer=process).fit_transform(dataset['text'])




# # Save the vectorizer
# dump(message, open("models/vectorizer.pkl", "wb"))

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(message, dataset['spam'], test_size=0.20, random_state=0)

# # Model creation
# model = MultinomialNB()

# # Model training
# model.fit(X_train, y_train)

# # Model saving
# dump(model, open("models/model.pkl", 'wb'))


# # Model predictions on test set
# y_pred = model.predict(X_test)


# # Model Evaluation | Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# accuracy * 100

# # Model Evaluation | Classification report
# classification_report(y_test, y_pred)


# # Model Evaluation | Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(dpi=100)
# sns.heatmap(cm, annot=True)
# plt.title("Confusion matrix")
# plt.show()



# import pickle

# vectorizer = pickle.load(open("/Users/saianish/Desktop/SpamML/models/vectorizer.pkl", "rb"))
# model = pickle.load(open("/Users/saianish/Desktop/SpamML/models/model.pkl", "rb"))

# # Example input
# sample_text = ["Win a free iPhone now! Click the link."]

# # Convert to features
# sample_features = vectorizer.transform(sample_text)

# # Predict spam or not
# prediction = model.predict(sample_features)
# print("Spam" if prediction[0] == 1 else "Not Spam")


import os
import ssl
import pickle
import nltk
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pickle import dump

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ensure SSL context for downloading NLTK data
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# Set NLTK data path
nltk.data.path.append('/Users/saianish/nltk_data')

# Ensure stopwords are downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir='/Users/saianish/nltk_data')
    stop_words = set(stopwords.words('english'))

# Define dataset path
dataset_path = "/Users/saianish/Desktop/SpamML/dataset/emails.csv"
if not os.path.exists(dataset_path):
    print(f"Error: Dataset not found at {dataset_path}")
    exit()

# Load dataset
dataset = pd.read_csv(dataset_path)

# Data cleaning: remove duplicates
dataset.drop_duplicates(inplace=True)

# Define text processing function
def process(text):
    nopunc = ''.join([char for char in text if char not in string.punctuation])
    clean = [word for word in nopunc.split() if word.lower() not in stop_words]
    return clean

# Create CountVectorizer
vectorizer = CountVectorizer(analyzer=process)
message = vectorizer.fit_transform(dataset['text'])

# Ensure models directory exists
model_dir = "/Users/saianish/Desktop/SpamML/models"
os.makedirs(model_dir, exist_ok=True)
vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
model_path = os.path.join(model_dir, "model.pkl")

# Save the vectorizer
dump(vectorizer, open(vectorizer_path, "wb"))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(message, dataset['spam'], test_size=0.20, random_state=0)

# Create and train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save trained model
dump(model, open(model_path, 'wb'))

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(dpi=100)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.title("Confusion Matrix")
plt.show()

# Load and test saved model
vectorizer = pickle.load(open(vectorizer_path, "rb"))
model = pickle.load(open(model_path, "rb"))

# Example prediction
# sample_text = ["Win a free iPhone now! Click the link."]
# sample_features = vectorizer.transform(sample_text)
# prediction = model.predict(sample_features)\
# print("Spam" if prediction[0] == 1 else "Not Spam")
