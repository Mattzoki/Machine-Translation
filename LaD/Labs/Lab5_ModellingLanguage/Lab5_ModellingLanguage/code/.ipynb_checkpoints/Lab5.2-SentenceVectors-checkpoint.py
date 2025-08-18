#!/usr/bin/env python
# coding: utf-8

# Copyright: Vrije Universiteit Amsterdam, Faculty of Humanities, CLTL

# # Lab 5.2: Bag-of-words

# In this lab, we see how sentences can be interpreted as a bag of words. 

# ## 1. Average word vectors

# A simple way to get a sentence representation is taking the average of its word vectors. This representation ignores word order though. We use the same model and data as in the previous lab. Adjust the data path to point to the correct folder. 

# In[1]:


from gensim.models import KeyedVectors
print("loading")
word2vec_model  = KeyedVectors.load_word2vec_format("../data/wiki-news-300d-1M.vec")
print("done loading")


# We can simply concatenate the vectors from all words, but this has the disadvantage that the number of dimensions of the sentence vector depends on the number of words. Alternatively, it is common to take the mean or the sum of the word vectors using the module *numpy*. **Make sure you understand [what it means](https://www.tutorialspoint.com/numpy/numpy_statistical_functions.htm) to calculate the mean of a vector along different axes.**   

# In[2]:


import numpy as np

# A function that takes a sentence as a list of tokens and returns the mean vector for a given model
def get_sentence_vector(sentence, model): 
    word_vectors =[]
    for word in sentence: 
        # Ignore words that are not in the vocabulary
        try: 
            index = model.key_to_index[word]
            word_vectors.append(model.get_vector(word))
        except:
            continue
    
    word_matrix = np.asarray(word_vectors)
    mean_vector = np.mean(word_matrix, axis=0)
    sum_vector = np.sum(word_matrix, axis=0)

    return mean_vector, sum_vector

sentence = ["What", "a", "lovely", "day"]

# The mean vector has the same number of dimensions (300) as the individual word vectors
print(get_sentence_vector(sentence, word2vec_model)[0].shape)


# Similarly as with word vectors, we can calculate the cosine similarity between two sentence vectors. However, conceptually this similarity is hard to interpret. 
# 
# **Think about the following questions: When are two sentences similar? How can we interpret the mean of words? Do you think this is a useful representation for sentences?** 

# In[3]:


from scipy.spatial import distance
sentence1 = ["Vegans", "just", "live", "longer"]
sentence2 = ["Veganism", "is", "a", "form", "of", "extremism"]
s1_vector = get_sentence_vector(sentence1, word2vec_model)[0]
s2_vector = get_sentence_vector(sentence2, word2vec_model)[0]
similarity = 1- distance.cosine(s1_vector,s2_vector)
print(sentence1)
print(sentence2)
print("Cosine Similarity: " + str(similarity))


#  **Play around with different sentences. Does it make a difference if we take the mean or the sum?**
#  
# You can apply pre-processing to the sentences before calculating the sentence vector. **Try out different pre-processing techniques (e.g. lemmatization, stop word removal, only keep most/least frequent words) and compare the effect on the similarity.**

# Now it is your turn: get the vectors for all sentences in your dataset. **The code below is incomplete and won't run correctly. Modify it to work with your dataset.** 

# In[9]:


import csv
file_path = '../../../Assignments/Assignment 2/Assignment 2/Final_df.csv'

newline_char = '\n'
csv_data = []

with open(file_path, "r", encoding="utf-8", newline=newline_char) as csvfile:
    reader = csv.reader(csvfile, delimiter=",")  # Use '\t' for TSV
    for row in reader:
        csv_data.append(row)

hindi_sents = []
english_sents = []
for i,row in enumerate(csv_data):
    hindi_s = row[2]
    english_s = row[1]
    hindi_sents.append(hindi_s)
    english_sents.append((i,english_s))

sent_id2vector ={}
sent_id2string ={}
# For article in your dataset:
for tupl in english_sents:
    sent_vector = get_sentence_vector(tupl[1],word2vec_model)
    sent_id2string [tupl[0]]= tupl[1]
    sent_id2vector[tupl[0]] = sent_vector
print(sent_id2vector[0])
# print(sent_id2string)


# Here is a little toy example how to calculate the similarity between all possible pairs. **Make sure you understand how to read the similarity matrix**. 

# In[10]:


from sklearn.metrics.pairwise import cosine_similarity

# Toy example: 
sentId2string ={"a1_s1":"Vegans just live longer", "a2_s1":"Veganism is a form of extremism","a3_s3":"Eating vegan is hard", "a4_s1":"How to build a car"  }

ids =[]
vectors =[]
for id, text in sentId2string.items():
    sentence_vector = get_sentence_vector(text.split(" "), word2vec_model)[0]
    ids.append(id)
    vectors.append(sentence_vector)

    
# Calculate the similarity between all possible pairs
similarity_matrix = cosine_similarity(np.asarray(vectors))
print(similarity_matrix)


# We can use *np.where* to retrieve only the matrix entries with a value above a certain threshold.

# In[11]:


threshold = 0.7
# We only want sentence pairs with a similarity above a threshold
result = np.where(similarity_matrix >threshold)
filtered_pairs= list(zip(result[0], result[1]))

for x,y in filtered_pairs:
    # We only look the lower triangle of the similarity matrix because it is symmetric 
    # and the values in the diagonal are always 1.
    if x < y:
        print("Highly similar pair: ")
       
        # x and y refer to the coordinates in the similarity matrix. We need to map them back to sentence ids
        print(ids[x], ids[y])
        print(sentId2string[ids[x]])
        print(sentId2string[ids[y]])
        print()


# ## 3. Plotting heatmaps

# Matrices can nicely be visualized using *seaborn*. **Check the [documentation](https://seaborn.pydata.org/generated/seaborn.heatmap.html).** 
# 
# If the heatmap is symmetric, it is good practice to only plot half of the matrix. **Change the plot to omit the mask=mask argument and take note of the difference.**

# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

# create a mask to hide the upper triangle
mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))
np.fill_diagonal(mask, False)

# check the difference between using the mask argument and omitting it
ax = sns.heatmap(similarity_matrix, mask=mask, linewidth=0.5, cmap="YlGnBu")


ax.set_xticklabels(ids)
ax.set_yticklabels(ids)
# Remove the ticks, but keep the labels
ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=True)
ax.set_title("Similarity between sentence pairs")
plt.show()


# ## 4. Document vectors in spacy
# 
# If you are using spacy as an nlp pipeline, you can directly get a vector for a document. Check the [documentation](https://spacy.io/usage/vectors-similarity). The document vector is the average over the vectors of the tokens. You need to double-check which pre-trained model is used to obtain the vectors. 

# In[13]:


import spacy

spacy_pipeline = spacy.load("en_core_web_sm")
doc = spacy_pipeline("Vegans just live longer")
print(doc.vector)
print(doc.vector.shape)


# In[14]:


doc = spacy_pipeline("Vegans just live longer and have a better life")
print(doc.vector)
print(doc.vector.shape)

