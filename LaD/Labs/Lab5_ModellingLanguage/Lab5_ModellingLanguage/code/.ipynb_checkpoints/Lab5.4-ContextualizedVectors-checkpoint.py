#!/usr/bin/env python
# coding: utf-8

# Copyright: Vrije Universiteit Amsterdam, Faculty of Humanities, CLTL

# # Lab 5.4: Contextualized Vectors

# Contextualized language models like *BERT* or *RoBERTa* provide different vectors for a token depending on its context. In the machine learning course, you learn more about the transformer architecture. In this lab, we learn how the same token is represented by different vectors depending on the context. When we want to compare sentences, we need to make sure that we compare two vectors with the same number of dimensions and not a list of vectors. Reducing a list of token vectors to a single sentence vector means losing information but it is necessary to compare the similarity between vectors.

# In[1]:


import torch
from transformers import AutoTokenizer, AutoModel
# Initialize roberta
model_name = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
    


# In[2]:


sentences = ["Vegans just live longer", "This is not just.", "I want just a bit of happiness."]

sentence_vectors = []
all_tokens = []
all_token_ids = []
for sentence in sentences: 
    # Use the bert tokenizer and add the special tokens (CLS, SEP)
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    all_tokens.append(tokens)
    
    # Tokens are automatically transformed into ids
    print(tokens)
    
    # Let's have a look at the subwords
    subwords = [str(tokenizer.decode([token])) for token in tokens]
    print(subwords)
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)

    # Get the bert output output
    model.eval()  # turn off dropout layers
    output = model(tokens_tensor)
 
    # The model provides a vector of 768 dimensions for each token
    # This vector corresponds to the last layer of hidden states of bert
    vector = output[0].detach().numpy()[0]
    print(type(vector))
    sentence_vectors.append(vector)
    print(vector.shape)
    print()
  
    


# **Make sure you understand the output.** You can see that the tokenizer uses subword units and special tokens at the beginning and the end of a sentence. 
# 
# You can see that "just" has the same token id (2074) in the three sentences. Its vector is however very different depending on the context, see code below. **Play around with different sentences and test tokens.** Interestingly, a token seems to be closer to other tokens in the same sentence than to the same token type in other sentences.  

# In[4]:


from scipy.spatial import distance
test_id = 95

# Get the position of the test token in the first sentence
position1 = all_tokens[0].index(test_id)
token_vector1 = sentence_vectors[0][position1]

# Get the position of the test token in the second sentence
position2 = all_tokens[1].index(test_id)
token_vector2 = sentence_vectors[1][position2]

# Calculate the similarity between the two token vectors
similarity = 1- distance.cosine(token_vector1,token_vector2)
print(similarity)


# It remains an open debate how to represent sentences with the same number of dimensions. Common approaches are to take the CLS token or the mean over the token representations. It is also an open question which layer produces good representations for which type of analysis.

# In[5]:


import numpy as np
print(sentence_vectors[0].shape)

# When taking the mean over tokens, we ignore the first (CLS) and the last (SEP) token.
tokens_only = np.array(sentence_vectors[0][1:-1])
print(tokens_only.shape)

# We take the mean
mean_sentence_vector = np.mean(tokens_only, axis =0)
print(mean_sentence_vector.shape)


# # Multilingual models
# Multilingual language models represent multiple languages in the same vector spaces. This facilitates cross-lingual analyses. We load multilingual BERT here. A more recent model is XLM-Roberta which you used in the [HLT course](https://github.com/cltl/ma-hlt-labs/blob/master/lab5.contextualized-models/How-to-use-bert-and-roberta.ipynb). You should be able to use it by changing the model_name to "xlm-roberta-base".

# In[6]:


model_name = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Have a look at the example sentences. **Note how the length of the sentences differs due to different subword splits.**

# In[7]:


sentences = ["I love deadlines.", "I procrastinate work.", "Ich liebe Deadlines.", "Ich prokrastiniere Arbeit."]
vectors = []
for sent in sentences:
    tokens = tokenizer.encode(sent, add_special_tokens=True)
    print([tokenizer.decode([token]) for token in tokens])
    model.eval()  # turn off dropout layers
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)
    output = model(tokens_tensor)
    vector = output[0].detach().numpy()[0]
    vectors.append(vector)
    print(vector.shape)
#calculate similarity (I guess we first have to take the mean over the vectors?)


# We again take the mean over tokens as a vector representation of the sentence and calculate the similarities between vectors. **Test different example sentences in various languages and analyse the results.** 

# In[8]:


from sklearn.metrics.pairwise import cosine_similarity
sentence_vectors = []
for vec in vectors: 
    token_vec = vec[1:-1]
    sentence_vec = np.mean(token_vec, axis =0)
    sentence_vectors.append(sentence_vec)
similarity_matrix = cosine_similarity(np.asarray(sentence_vectors))
print(similarity_matrix)

