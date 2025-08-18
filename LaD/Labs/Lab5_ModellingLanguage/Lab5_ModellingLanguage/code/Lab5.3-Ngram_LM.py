#!/usr/bin/env python
# coding: utf-8

# Copyright: Vrije Universiteit Amsterdam, Faculty of Humanities, CLTL
# 
# # 5.3 Ngram Language Modelling with NLTK
# 
# In this lab, we learn how to build an n-gram based language model to calculate surprisal.

# ## 1. Preprocessing
# First we load a tokenizer and a [text](https://gist.githubusercontent.com/alvations/53b01e4076573fea47c6057120bb017a/raw/b01ff96a5f76848450e648f35da6497ca9454e4a/language-never-random.txt). **Have a look at the characteristics of the text. How is hyphenated text treated by the tokenizer?**

# In[16]:


get_ipython().system('pip install -U pip')
get_ipython().system('pip install -U dill')
get_ipython().system('pip install -U nltk==3.4')


# In[17]:


try: # Use the default NLTK tokenizer.
    from nltk import word_tokenize, sent_tokenize 
    # Testing whether it works. 
    # Sometimes it doesn't work on some machines because of setup issues.
    word_tokenize(sent_tokenize("This is a test sentence. Yes it is.")[0])
except: # Use a naive sentence tokenizer and the toktok tokenizer.
    import re
    from nltk.tokenize import ToktokTokenizer
    # See https://stackoverflow.com/a/25736515/610569
    sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)
    # Use the toktok tokenizer that requires no dependencies.
    toktok = ToktokTokenizer()
    word_tokenize = toktok.tokenize


# In[18]:


import os
import requests
import io 

# Text version of https://kilgarriff.co.uk/Publications/2005-K-lineer.pdf
if os.path.isfile('language-never-random.txt'):
    with io.open('language-never-random.txt', encoding='utf8') as fin:
        text = fin.read()
else:
    url = "https://gist.githubusercontent.com/alvations/53b01e4076573fea47c6057120bb017a/raw/b01ff96a5f76848450e648f35da6497ca9454e4a/language-never-random.txt"
    text = requests.get(url).content.decode('utf8')
    with io.open('language-never-random.txt', 'w', encoding='utf8') as fout:
        fout.write(text)

# Tokenize the text.
tokenized_text = [list(map(str.lower, word_tokenize(sent))) 
                  for sent in sent_tokenize(text)]
print(tokenized_text[0:3])

# Flatten the list (notice that the list comprehension is quite complex)
all_tokens = [tok for sent in tokenized_text for tok in sent]

num_tokens = len(all_tokens)
num_sentences = len(tokenized_text)
print("Tokens", num_tokens)
print("Sentences", num_sentences)


# Then we preprocess the text. We use the *padded_everygram_pipeline* function from nltk. It pads the sentence and then makes so-called *everygrams*. Let's see what this looks like. **Make sure you understand what is meant by everygrams here. How does the padding change when you change n? How dow the n-grams change if you change n?**

# In[19]:


from nltk.lm.preprocessing import padded_everygram_pipeline
# Set the n-gram size
n = 3

# Let us check what is happening on a subset: 
ngram_data, padded = padded_everygram_pipeline(n, tokenized_text[0:3])

# What is happening during padding? 
print("PADDING:")
print(list(padded))

# What kind of ngrams do we get? 
print("\n\nNGRAMS:")
for ngrams in ngram_data: 
    print(list(ngrams))
    print()
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)




# ## 2. Training the N-gram Model

# Having prepared our data we are ready to start training a model. As a simple example, let us train a Maximum Likelihood Estimator (MLE).
# 
# We only need to specify the highest ngram order to instantiate it. **Note that train_data is an iterator object. If you want to execute the next cell twice, you first need to re-run the previous one.** (Make sure the vocab has more than 0 items in the second print statement.)

# In[20]:


from nltk.lm import MLE
model = MLE(n) # Lets train a trigram model, previously we set n=3

# We fit the model, filling the vocabulary on the way
print(model.vocab)
model.fit(train_data, padded_sents)
print(model.vocab)



# # 3. Using the N-gram Language Model
# An ngram-based language model simply counts the frequency of n-grams during training and stores them in an easily accessible counter representation. 

# In[21]:


print(model.counts)
# counts for unigrams:
print(model.counts['language']) # i.e. Count('language')
# count for bigrams
print(model.counts[['language']]['is']) # i.e. Count('is'|'language')
# count for trigrams
print(model.counts[['language', 'is']]['never']) # i.e. Count('never'|'language is')


# Based on the ngram-frequencies, it calculates the probability of a word in a certain context. 
# For the MLE model, the probability is calculated as the relative frequency. **Note that the model score is lower than the probability.** This has to do with the padding tokens. **Make sure you understand how the score is adjusted.**
# 

# In[22]:


model_score = model.score('language')
probability = model.counts['language']/num_tokens
                                    

print("\nProbability of the word 'language'")
print("{:.5f}".format(model_score))
print("{:.5f}".format(probability))

print("\nAdjust for padding tokens")
all_padding_tokens = num_sentences * (n-1) * 2
print(num_tokens, all_padding_tokens)
adjusted_probability = model.counts['language']/(num_tokens + all_padding_tokens)
print("{:.5f}".format(adjusted_probability))

print("\nProbabilities padding tokens")
print("{:.5f}".format(model.score('<s>')))
print("{:.5f}".format(model.score('</s>')))


# We can also calculate the frequency for higher n-grams. 

# In[23]:


# bigram
print(model.score('is', ['language']))  # P('is'|'language')
# trigram
print(model.score('never', ['language', 'is']))  # P('never'|'language is')
print()

# 4-grams don't work
print(model.score('choose', ['language','users','never'])) 
print(model.score('never', ['language','users'])) 
# Language users only occurs once in the text


# In[24]:


# To avoid underflow when working with many small score values, we usually work with log probabilities instead. 
# This can be done with the `logscore` method.
print(model.score('enormous', ['is']))
print(model.logscore('enormous', ['is']))



# ## 4. The Unknown Word Problem

# In[25]:


# The vocabulary helps us handle words that have not occurred during training.
# If we lookup the vocab on unseen sentences not from the training data, 
# it automatically replace words not in the vocabulary with `<UNK>`.
print(model.vocab.lookup('language is never blah and .'.split()))


# The standard MLE language model has a problem with unknown words. 

# In[26]:


# Items that are not seen during training are mapped to the vocabulary's "unknown label" token.  This is "<UNK>" by default.
print(model.score("<UNK>") == model.score("blah"))

# The MLE model does not apply any smoothing, so the probability for UNK is 0
print(model.score("<UNK>"))

# As a consequence, the probability for a phrase containing an unknown word is also 0. 
print(model.score('blah', ['language', 'is']))


# This problem can be solved with more advanced smoothing techniques, for example Laplace smoothing.

# In[27]:


from nltk.lm import Laplace
n = 3
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
smoothed_model_small =  Laplace(n)
smoothed_model_small.fit(train_data, padded_sents)
print(smoothed_model_small.score('language'))
print(smoothed_model_small.score('is'))
print(smoothed_model_small.score('blah', ['language', 'is']))


# ## 5. Surprisal

# In order to calculate surprisal for a sentence, we should first train a model on a bigger corpus. **Training the model takes a while**. Make sure you have downloaded the Brown corpus: 
# 
# *If you get a certificate error, you can download the Brown corpus in the terminal and/or check this [https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed](thread). 

# In[28]:


import nltk
nltk.download('brown')


# In[29]:


from nltk.corpus import brown

#print(brown.words())
# Train a bigram model on the brown corpus
n = 2
corpus = " ".join([word.lower() for word in brown.words()])

tokenized_corpus = [list(map(str.lower, word_tokenize(sent))) 
                  for sent in sent_tokenize(corpus)]
len(tokenized_corpus)
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_corpus)

smoothed_model =  Laplace(n)
smoothed_model.fit(train_data, padded_sents)
print(smoothed_model.vocab)


# Try out a few sentences. You will notice that the Brown corpus is still way too small for informative probabilities. Especially if you want to increase the n-gram size, you need more data. 

# In[30]:


import math
example = "The personnel is lacking experience. The sneedleweedle is gobberwuming experience."
tokenized_example = [list(map(str.lower, word_tokenize(sent))) 
                  for sent in sent_tokenize(example)]
ngram_data, _ = padded_everygram_pipeline(n, tokenized_example)

for ngrams in ngram_data:

    max_ngrams = [ngram for ngram in ngrams if len(ngram)==n]
    # We ignore the first and last ngram because the padding tokens skew the probabilities too much
    max_grams = max_ngrams[1:-1]
    print(max_grams)
    sentence_prob = 1
    for ngram in max_grams:
        # Split the ngram into the last word and the preceeding context
        word = ngram[n-1]
        context = list(ngram[0:(n-1)])
        # Get the probability for the n-gram
        ngram_prob = smoothed_model.score(word, context)
        
        # Inspect the probabilities of the individual n-grams
#         print(ngram)
#         print("{:.5f}".format(ngram_prob))
        # Update the sentence probability
        sentence_prob *= ngram_prob
    print(sentence_prob)
    surprisal = -math.log(sentence_prob)
    print(surprisal)

