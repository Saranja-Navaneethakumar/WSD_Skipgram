from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

import numpy as np
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

path = '/content/drive/MyDrive/4th year/Research/codes/datasets/semeval-2017-train.csv'
train_dataset = pd.read_csv(path, delimiter='	')
train_dataset.columns = ['Label', 'Text']
train_dataset.rename(columns={'label': 'Label','text' : 'Text'})


stop_words = set(stopwords.words('english'))

# Preprocess text by removing stopwords and tokenizing
def preprocess_text(train_dataset):
    tokens = word_tokenize(train_dataset.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

text_column = 'Text'  # Replace 'text' with the actual column name in your CSV file
sentences = train_dataset[text_column].tolist()

# Display the sentences
# for sentence in sentences:
#     print(sentence)

test_dataset_path = "/content/drive/MyDrive/4th year/Research/Dataset/testdata.xlsx - Sheet1.csv"
test_dataset = pd.read_csv(test_dataset_path)
test_dataset.columns = ['ID', 'Sentence', 'Polysemy_Word']
test_dataset.rename(columns={'id': 'ID','sentence/context': 'Sentence','polysemy_word' : 'Polysemy_Word'})

test_text_column = 'Sentence'  # Replace 'Text' with the actual column name in your dataset
test_sentences = test_dataset['Sentence'].tolist()

# Combine training and labeled datasets
combined_dataset = sentences + test_sentences

# Preprocess the combined dataset
preprocessed_dataset = [preprocess_text(instance) for instance in combined_dataset]

# Train the Word2Vec model
model = Word2Vec(preprocessed_dataset, sg=1, window=5, min_count=1, vector_size=100)

# Get word vectors
word_vectors = model.wv

# Perform word sense disambiguation
def word_sense_disambiguation(word):
    # Get the most similar words
    similar_words = word_vectors.most_similar(positive=[word])
    return similar_words

# Calculate accuracy
correct_predictions = 0
total_instances = len(test_dataset)

for instance in test_dataset.itertuples():
    sentence = instance.Sentence
    target_word = instance.Polysemy_Word
    label = instance.Polysemy_Word 

    # Skip instances where the target word is not in the vocabulary
    if target_word not in word_vectors:
        continue

    predicted_label = word_sense_disambiguation(target_word)[0][0]
    if predicted_label == label:
        correct_predictions += 1

accuracy = (correct_predictions / total_instances) * 100
print(f"Accuracy: {accuracy}%")

