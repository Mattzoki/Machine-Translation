#!/usr/bin/env python
# coding: utf-8

# Copyright: Vrije Universiteit Amsterdam, Faculty of Humanities, CLTL

# # Lab 5.1: Querying Word Vectors
# 
# In this lab we learn how to represent words using a pre-trained model. We will work with the English fasttext model [wiki-news-300d-1M.vec](https://fasttext.cc/docs/en/english-vectors.html). Download it and save it to the data folder. 
# 
# You can download models for other languages [here](https://fasttext.cc/docs/en/crawl-vectors.html) at the bottom of the page. 
# 
# We use the *gensim* module to query the model. The model is quite big, so the following loading step takes long: 

# In[1]:


from gensim.models import KeyedVectors
print("loading")
word_vectors = KeyedVectors.load_word2vec_format("../data/wiki-news-300d-1M.vec")
print("done loading")


# We can now get the vector representation for a word from the model. Note that the model only contains one million words. Some words can thus not be found and you will get a *KeyError*.  

# In[2]:


term ="veganism"
term_vector = word_vectors.get_vector(term)

print(term_vector)


# ## 1. Calculating similarity
# Gensim provides different options to calculate the similarity between words. The standard measure for determining the similarity between words is to calculate the cosine similarity between their vectors. 

# In[3]:


term1 ="vegetables"
term2 = "fruit"

word_vectors.similarity(term1,term2)


# In[4]:


word_vectors.most_similar(term)


# ## 2. Arithmetic operations
# 
# Word vectors have become famous because they make it possible to perform arithmetic operations over word vectors. The most popular example is: 
# woman + king - man = queen

# In[5]:


word_vectors.most_similar(positive=["woman", "king"], negative=["man"])


# In[6]:


term ="veganism"
subtract ="vegetarian"
word_vectors.most_similar(positive=[term], negative=[subtract])


# ## 3. Training your own model
# 
# The word vectors have been trained on Wikipedia. You can also train word vectors on your own dataset. **In this example, we only apply tokenization and lowercasing. Discuss which pre-processing steps you want to use for your dataset.**

# In[7]:


import pandas as pd
import stanza

# Read in TSV
tsv_file = "../data/veganism_overview_en.tsv"
content = pd.read_csv(tsv_file, sep="\t", header = 0, keep_default_na=False)
articles = content["Text"]
stanza.download("en")
tokenizer = stanza.Pipeline('en', processors='tokenize')
tokenized = []
for article in articles:
    for sent in tokenizer(article).sentences:
        tokenized.append([tok.text.lower() for tok in sent.tokens])
print(tokenized)


# In[8]:


from gensim.models import Word2Vec
# Train a Word2Vec model, the min_count parameter indicates the minimum frequency of each word in the corpus
mymodel = Word2Vec(tokenized, min_count=2)

# summarize the loaded model
print(mymodel)


# Let's have a look at the 50 most frequent words
for index in range(50):
    print(mymodel.wv.index_to_key[index])


# In[9]:


mymodel.wv.most_similar("plant")


# You can see that function words are very dominant in this model because the dataset is so small. Training own word vectors is only reasonable with much larger datasets. **Discuss what would be a good training corpus for your analysis goals.**

# ## 4. Visualizing word vectors
# 
# Word vectors with 100 dimensions are difficult to conceptualize for human brains. As a simplification, it is common to apply a dimensionality reduction technique on the word vector to obtain a two-dimensional representation. Obviously, this two-dimensional vector contains much less information than the 100-dimensional vector. However, it can yield nice visualizations to provide anecdotal evidence.  
# 
# Popular algorithm for calculating the reduced dimensions are principal component analysis (PCA) and [T-SNE](https://lvdmaaten.github.io/tsne/). 
# 
# **Play around with different models, reduction techniques and term selection. Note that the dimensionality reduction for the big fasttext model takes very long and you might run into memory errors. It could help to run the code in a regular python file (.py) instead of in a jupyter notebook.** [This article](https://towardsdatascience.com/why-you-are-using-t-sne-wrong-502412aab0c0) provides some good advice on interpreting T-SNE output. 
# 
# 

# In[10]:


from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
import numpy as np
# Select model 
model=word_vectors

# If you use a large model, you need to restrict the vocabulary to the most frequent terms
# If you use the full vocab it might cause a crash of the notebook
# Restricted
vocab = [model.index_to_key[i] for i in range(8000)]

# Apply dimensionality reduction with PCA or T-SNE
high_dimensional = np.asarray([model[w] for w in vocab])
reduction_technique = TSNE(n_components=2)
#reduction_technique = PCA(n_components=2)

print("Calculate dimensionality reduction")
two_dimensional = reduction_technique.fit_transform(high_dimensional)
print("Done")


# In[11]:


# Get the indices in the vocabulary for selected terms
terms =["fruit", "meat", "salt","eat", "healthy", "good", "sugar", "sweet"]
term_indices = [model.key_to_index[t] for t in terms]

print(term_indices)


# In[12]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize = (15, 10))

# Plot the two-dimensional vectors for the selected terms
x_values = [two_dimensional[index, 0] for index in term_indices]
y_values = [two_dimensional[index, 1] for index in term_indices]

ax.plot(x_values, y_values, 'o')


# Let's improve the plot a bit. **Play around with the plotting options.**

# In[13]:


import matplotlib.cm as cm
import numpy as np 
fig, ax = plt.subplots(1, 1, figsize = (15, 10))

colors = cm.rainbow(np.linspace(0, 1, len(terms)))
for x, y, c in zip(x_values, y_values, colors):
    ax.plot(x, y, 'o', markersize=12, color=c)

# Add title and description
ax.set_title('My Terms')
description="The word vectors for selected terms in the English fasttext model (wiki-news-300d-1M.vec) reduced to two dimensions using the t-sne algorithm. "
fig.text(.51, .05, description, ha="center", fontsize=12)

# Hide the ticks 
ax.set_yticks([]) 
ax.set_xticks([])

# Annotate the terms in the plot
for i, word in enumerate(terms):
	plt.annotate(word, xy=(x_values[i], y_values[i]), fontsize = 12)
    
plt.show()


