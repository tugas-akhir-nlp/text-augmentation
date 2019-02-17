
# coding: utf-8

# In[13]:


import timeit
import json


# In[7]:


with open('corpus_path.txt', 'r') as file:
    path = file.readlines()[0].strip()
print(path)


# In[15]:


start = timeit.default_timer()

with open(path) as file:
    corpus = file.readlines()
    
lm = {}
lm['unigram'] = {}
lm['unigram']['_count_'] = 0
lm['bigram'] = {}
lm['trigram'] = {}

prev1 = None
for kalimat in corpus:
    seq = kalimat.strip()
    seq = seq.split(' ')
    
    prev1 = None
    for word in seq:
        # unigram processing
        lm['unigram']['_count_'] += 1
        if word not in lm['unigram']:
            lm['unigram'][word] = 1
        else:
            lm['unigram'][word] += 1
            
        # bigram processing
        if prev1 != None:
            if prev1 not in lm['bigram']:
                lm['bigram'][prev1] = {}
                lm['bigram'][prev1]['_count_'] = 0
            lm['bigram'][prev1]['_count_'] += 1
            if word not in lm['bigram'][prev1]:
                lm['bigram'][prev1][word] = 1
            else:
                lm['bigram'][prev1][word] += 1
        
        # trigram processing
        if prev1 != None and prev2 != None:
            if prev2 not in lm['trigram']:
                lm['trigram'][prev2] = {}
#                 lm['trigram'][prev2]['_count_'] = 0
            if prev1 not in lm['trigram'][prev2]:
                lm['trigram'][prev2][prev1] = {}
                lm['trigram'][prev2][prev1]['_count_'] = 0
            lm['trigram'][prev2][prev1]['_count_'] += 1
            if word not in lm['trigram'][prev2][prev1]:
                lm['trigram'][prev2][prev1][word] = 1
            else:
                lm['trigram'][prev2][prev1][word] += 1
            
            
        # assign
        prev2 = prev1
        prev1 = word
                
stop = timeit.default_timer()
execution_time = stop - start
print('execution time: {}'.format(execution_time))

with open('statistical_lm.json', 'w') as file:
    json.dump(lm, file)