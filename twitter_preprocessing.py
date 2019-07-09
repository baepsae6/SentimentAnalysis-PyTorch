#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import csv


# In[2]:


data_pos = pd.read_csv('positive.csv', delimiter=';', header=None)
data_neg = pd.read_csv('negative.csv', delimiter=';', header=None)


# In[ ]:





# In[3]:


data_pos = data_pos[3]
data_pos = pd.DataFrame(data_pos)
data_pos['labels'] = 1
data_pos = data_pos.rename(index=int, columns={3: "text"})


# In[4]:


data_neg.head()


# In[5]:


data_neg = data_neg[3]
data_neg = pd.DataFrame(data_neg)
data_neg['labels'] = 0
data_neg = data_neg.rename(index=int, columns={3: "text"})


# In[6]:


import nltk
nltk.download('punkt')
from nltk import (
    sent_tokenize as splitter,
    wordpunct_tokenize as tokenizer
)
def tokenize(text):
    return [tokenizer(sentence) for sentence in splitter(text)]

def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

def tokenize_flatten_df(row, field):
    return flatten(tokenize(row[field]))


# In[7]:


import re

def remove_urls(text):
    return re.sub(r"(https?\://)\S+", "", text)

def remove_mentions(text):
    return re.sub(r"@[^:| ]+:? ?", "", text)

def remove_rt(text):
    if text.lower().startswith("rt"):
        return text[3:].strip()
    return text

def remove_urls_mentions_rt_df(row, field):
    return remove_rt(remove_mentions(remove_urls(row[field])))


# In[8]:


from nltk.corpus import stopwords
nltk.download('stopwords')

def replace_hashtags_from_list(tokens_list):
    return [token for token in tokens_list if token != "#"]

def remove_digits(tokens_list):
    return [token for token in tokens_list 
                if not re.match(r"[-+]?\d+(\.[0-9]*)?$", token)]

def remove_containing_non_alphanum(tokens_list):
    return [re.sub(r'[^а-яА-Я\(\)\:]', "", token) for token in tokens_list]
                
def lowercase_list(tokens_list):
    return [token.lower() for token in tokens_list]

def remove_stopwords(tokens_list):
    return [token for token in tokens_list
                if not token in stopwords.words(u'russian')]

def clean_tokens(row, field):
    return replace_hashtags_from_list(
        remove_digits(
            remove_containing_non_alphanum(
                lowercase_list(remove_stopwords(row[field])))))


# In[9]:


data_pos['text_cleaned_from_url_mentions_rt'] =     data_pos.apply(
        lambda row: remove_urls_mentions_rt_df (row, 'text'),
        axis=1)

data_pos['text_tokenized'] =     data_pos.apply(
        lambda row:
            tokenize_flatten_df (row, 'text_cleaned_from_url_mentions_rt'),
        axis=1)

data_pos['text_tokenized_cleaned'] =     data_pos.apply(
        lambda row:
            clean_tokens (row, 'text_tokenized'),
        axis=1)


# In[ ]:


data_neg['text_cleaned_from_url_mentions_rt'] =     data_neg.apply(
        lambda row: remove_urls_mentions_rt_df (row, 'text'),
        axis=1)

data_neg['text_tokenized'] =     data_neg.apply(
        lambda row:
            tokenize_flatten_df (row, 'text_cleaned_from_url_mentions_rt'),
        axis=1)

data_neg['text_tokenized_cleaned'] =     data_neg.apply(
        lambda row:
            clean_tokens (row, 'text_tokenized'),
        axis=1)


# In[ ]:


print(data_pos.shape)
print(data_neg.shape)


# In[ ]:


data = pd.concat([data_pos, data_neg], ignore_index=True)
data


# In[ ]:


data = data.drop(['text', 'text_tokenized', 'text_cleaned_from_url_mentions_rt'], axis=1)


# In[ ]:


data.columns = ['label', 'text']


# In[ ]:


data = data[data['text'].map(lambda d: len(d)) > 0]


# In[ ]:


data = data.to_dict()


# In[ ]:


def write_json(file_name, data):
    with open(file_name + '.json', 'w') as f:
        json.dump(data, f)


# In[ ]:


#write_json('twitter_prep_data', data)


# In[ ]:


# def write_csv(filename, data):
#     with open(filename + '.csv', 'w') as f:
#         writer = csv.writer(f)
#         for row in data.values:
#             writer.writerow(row)
        
#     f.close()


# In[ ]:


# write_csv('twitter_prep_data', data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




